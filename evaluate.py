"""
체크포인트에서 모델을 로드하여 평가하는 스크립트
"""
import argparse
import torch
import torch.nn as nn
import os
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from src.model import Transformer
from src.data_utils import create_tokenizer, create_token_based_data_loader, prepare_sample_data, load_tokenizer
from src.trainer import LabelSmoothingLoss

class ModelEvaluator:
    def __init__(self, checkpoint_path, device='auto'):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model = None
        self.config = None
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.criterion = None
        
        print(f"Evaluator initialized with device: {self.device}")
        
    def load_checkpoint(self):
        """체크포인트 로드"""
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        print(f"Checkpoint info:")
        print(f"  - Config: {self.config.get('description', 'Custom config')}")
        if 'step' in checkpoint:
            print(f"  - Training step: {checkpoint['step']}")
        if 'val_loss' in checkpoint:
            print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint
    
    def load_tokenizers(self):
        """토크나이저 로드"""
        print("Loading tokenizers...")
        
        src_tokenizer_path = "tokenizers/src_tokenizer.json"
        tgt_tokenizer_path = "tokenizers/tgt_tokenizer.json"
        
        if os.path.exists(src_tokenizer_path) and os.path.exists(tgt_tokenizer_path):
            self.src_tokenizer = load_tokenizer(src_tokenizer_path)
            self.tgt_tokenizer = load_tokenizer(tgt_tokenizer_path)
            print(f"Loaded tokenizers from saved files")
        else:
            print("Saved tokenizers not found. Creating new ones...")
            # 데이터에서 토크나이저 재생성
            src_texts, tgt_texts = prepare_sample_data()
            data_config = self.config['data']
            
            # 학습 시와 동일한 데이터로 토크나이저 생성
            train_src = src_texts * data_config['data_multiplier']
            train_tgt = tgt_texts * data_config['data_multiplier']
            
            self.src_tokenizer = create_tokenizer(train_src, vocab_size=data_config['vocab_size'])
            self.tgt_tokenizer = create_tokenizer(train_tgt, vocab_size=data_config['vocab_size'])
        
        print(f"Source vocabulary size: {self.src_tokenizer.get_vocab_size()}")
        print(f"Target vocabulary size: {self.tgt_tokenizer.get_vocab_size()}")
    
    def build_model(self, checkpoint):
        """모델 구성 및 가중치 로드"""
        print("Building and loading model...")
        
        model_config = self.config['model']
        
        self.model = Transformer(
            src_vocab_size=self.src_tokenizer.get_vocab_size(),
            tgt_vocab_size=self.tgt_tokenizer.get_vocab_size(),
            d_model=model_config['d_model'],
            n_heads=model_config['h'],
            n_layers=model_config['N'],
            d_ff=model_config['d_ff'],
            max_seq_length=model_config['max_seq_length']
        ).to(self.device)
        
        # 드롭아웃 설정
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = model_config['P_drop']
        
        # 모델 가중치 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded successfully")
        print(f"Total parameters: {total_params:,}")
        print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")
        
        # 손실 함수 설정
        training_config = self.config['training']
        self.criterion = LabelSmoothingLoss(
            self.tgt_tokenizer.get_vocab_size(),
            smoothing=training_config['label_smoothing'],
            ignore_index=0
        )
    
    def prepare_data(self, data_type='validation'):
        """평가용 데이터 준비"""
        print(f"Preparing {data_type} data...")
        
        src_texts, tgt_texts = prepare_sample_data()
        data_config = self.config['data']
        
        if data_type == 'validation':
            # 원본 데이터를 검증용으로 사용
            eval_src = src_texts
            eval_tgt = tgt_texts
        elif data_type == 'train':
            # 확장된 학습 데이터로 평가
            eval_src = src_texts * data_config['data_multiplier']
            eval_tgt = tgt_texts * data_config['data_multiplier']
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")
        
        # 토큰 기반 데이터로더 생성
        batch_tokens = self.config['training']['batch_tokens']
        max_length = data_config['max_length']
        
        self.eval_loader = create_token_based_data_loader(
            eval_src, eval_tgt, self.src_tokenizer, self.tgt_tokenizer,
            batch_tokens=batch_tokens, max_length=max_length, shuffle=False
        )
        
        print(f"Data prepared: {data_type} set with {len(eval_src)} samples")
        
        # 샘플 배치 정보
        sample_batch = next(iter(self.eval_loader))
        src_tokens = (sample_batch['src'] != 0).sum().item()
        tgt_tokens = (sample_batch['tgt_input'] != 0).sum().item()
        total_tokens = src_tokens + tgt_tokens
        
        print(f"Sample batch info:")
        print(f"  - Batch size (sentences): {sample_batch['src'].size(0)}")
        print(f"  - Total tokens in batch: {total_tokens}")
        print(f"  - Target batch tokens: {batch_tokens}")
    
    def evaluate_full(self, max_batches=None):
        """전체 데이터에 대한 상세 평가"""
        print("\nStarting full evaluation...")
        
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        num_batches = 0
        batch_losses = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.eval_loader, desc="Evaluating")
            
            for batch_idx, batch in enumerate(progress_bar):
                if max_batches and batch_idx >= max_batches:
                    break
                
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
                loss = self.criterion(output, tgt_output)
                
                # 토큰 수 계산 (패딩 제외)
                batch_tokens = (tgt_output != 0).sum().item()
                
                total_loss += loss.item()
                total_tokens += batch_tokens
                num_batches += 1
                batch_losses.append(loss.item())
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}',
                    'tokens': batch_tokens
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # 통계 계산
        batch_losses = np.array(batch_losses)
        loss_std = np.std(batch_losses)
        loss_min = np.min(batch_losses)
        loss_max = np.max(batch_losses)
        
        results = {
            'average_loss': avg_loss,
            'perplexity': perplexity,
            'total_batches': num_batches,
            'total_tokens': total_tokens,
            'loss_std': loss_std,
            'loss_min': loss_min,
            'loss_max': loss_max,
            'batch_losses': batch_losses.tolist()
        }
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Total Batches: {num_batches}")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"Loss Std: {loss_std:.4f}")
        print(f"Loss Range: [{loss_min:.4f}, {loss_max:.4f}]")
        print("="*60)
        
        return results
    
    def evaluate_samples(self, num_samples=5):
        """몇 개 샘플에 대한 상세 분석"""
        print(f"\nEvaluating {num_samples} sample translations...")
        
        self.model.eval()
        samples_evaluated = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
                predictions = torch.argmax(output, dim=-1)
                
                batch_size = src.size(0)
                for i in range(min(batch_size, num_samples - samples_evaluated)):
                    print(f"\n--- Sample {samples_evaluated + 1} ---")
                    
                    # 소스 문장
                    src_tokens = src[i].cpu().numpy()
                    src_tokens = src_tokens[src_tokens != 0]  # 패딩 제거
                    src_text = self.src_tokenizer.decode(src_tokens.tolist())
                    print(f"Source: {src_text}")
                    
                    # 실제 타겟
                    tgt_tokens = tgt_output[i].cpu().numpy()
                    tgt_tokens = tgt_tokens[tgt_tokens != 0]  # 패딩 제거
                    tgt_text = self.tgt_tokenizer.decode(tgt_tokens.tolist())
                    print(f"Target: {tgt_text}")
                    
                    # 예측
                    pred_tokens = predictions[i].cpu().numpy()
                    pred_tokens = pred_tokens[pred_tokens != 0]  # 패딩 제거
                    pred_text = self.tgt_tokenizer.decode(pred_tokens.tolist())
                    print(f"Prediction: {pred_text}")
                    
                    # 토큰 레벨 정확도
                    tgt_seq = tgt_output[i]
                    pred_seq = predictions[i]
                    mask = (tgt_seq != 0)
                    if mask.sum() > 0:
                        accuracy = (tgt_seq[mask] == pred_seq[mask]).float().mean().item()
                        print(f"Token Accuracy: {accuracy:.4f}")
                    
                    samples_evaluated += 1
                
                if samples_evaluated >= num_samples:
                    break
        
        print(f"\nSample evaluation completed ({samples_evaluated} samples)")
    
    def save_results(self, results, output_dir):
        """평가 결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON 결과 저장
        results_with_metadata = {
            'checkpoint_path': self.checkpoint_path,
            'evaluation_time': datetime.now().isoformat(),
            'config': self.config,
            'device': str(self.device),
            'results': results
        }
        
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            # batch_losses는 너무 길 수 있으므로 별도 처리
            save_results = results_with_metadata.copy()
            batch_losses = save_results['results'].pop('batch_losses', [])
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        
        # 배치별 손실 그래프 저장
        if batch_losses:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(batch_losses, alpha=0.7)
            plt.axhline(y=results['average_loss'], color='r', linestyle='--', 
                       label=f'Average: {results["average_loss"]:.4f}')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title('Loss per Batch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(batch_losses, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(x=results['average_loss'], color='r', linestyle='--', 
                       label=f'Average: {results["average_loss"]:.4f}')
            plt.xlabel('Loss')
            plt.ylabel('Frequency')
            plt.title('Loss Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'loss_analysis.png'), dpi=300)
            plt.close()
        
        print(f"Results saved to: {output_dir}")
        print(f"  - evaluation_results.json")
        print(f"  - loss_analysis.png")

def main():
    parser = argparse.ArgumentParser(description='체크포인트에서 모델 평가')
    parser.add_argument('checkpoint', type=str, help='체크포인트 파일 경로')
    parser.add_argument('--data_type', type=str, default='validation', 
                       choices=['validation', 'train'], help='평가할 데이터 타입')
    parser.add_argument('--max_batches', type=int, default=None, 
                       help='최대 평가 배치 수 (None이면 전체)')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='상세 분석할 샘플 수')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='결과 저장 디렉토리')
    parser.add_argument('--no_samples', action='store_true', 
                       help='샘플 분석 생략')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"evaluation_{checkpoint_name}_{timestamp}"
    
    print("Transformer 모델 평가 시작")
    print("=" * 60)
    print(f"체크포인트: {args.checkpoint}")
    print(f"데이터 타입: {args.data_type}")
    print(f"최대 배치: {args.max_batches or 'All'}")
    print(f"출력 디렉토리: {args.output_dir}")
    print("=" * 60)
    
    # 평가자 생성
    evaluator = ModelEvaluator(args.checkpoint)
    
    # 체크포인트 로드
    checkpoint = evaluator.load_checkpoint()
    
    # 토크나이저 로드
    evaluator.load_tokenizers()
    
    # 모델 구성
    evaluator.build_model(checkpoint)
    
    # 데이터 준비
    evaluator.prepare_data(args.data_type)
    
    # 전체 평가
    results = evaluator.evaluate_full(args.max_batches)
    
    # 샘플 분석
    if not args.no_samples:
        evaluator.evaluate_samples(args.num_samples)
    
    # 결과 저장
    evaluator.save_results(results, args.output_dir)
    
    print(f"\n평가 완료! 결과는 {args.output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
