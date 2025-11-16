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
from src.data_utils import create_tokenizer, create_token_based_data_loader, load_tokenizer
from src.trainer import LabelSmoothingLoss
from src.metrics import EvaluationMetrics, batch_decode_for_evaluation
from src.bpe_adapter import load_bpe_tokenizers, create_bpe_token_based_data_loader, save_bpe_tokenizers
from src.data_loader import load_problem_data, clean_sentence_pairs

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
        """BPE 토크나이저 로드 (trainer와 동일한 방식)"""
        print("Loading BPE tokenizers...")
        
        src_model_path = "tokenizers/src_bpe.model"
        tgt_model_path = "tokenizers/tgt_bpe.model"
        
        if os.path.exists(src_model_path) and os.path.exists(tgt_model_path):
            self.src_tokenizer, self.tgt_tokenizer = load_bpe_tokenizers()
            print(f"✓ Loaded BPE tokenizers from saved model files")
        else:
            print("⚠️  Saved BPE tokenizers not found. Creating new BPE tokenizers...")
            from src.bpe_adapter import create_bpe_tokenizers
            
            # 새로운 BPE 토크나이저 생성 (trainer와 동일한 방식)
            self.src_tokenizer, self.tgt_tokenizer = create_bpe_tokenizers(self.config)
            
            # 토크나이저 저장 (trainer와 동일한 방식)
            save_bpe_tokenizers(self.src_tokenizer, self.tgt_tokenizer)
            print(f"✓ Created and saved new BPE tokenizers")
        
        print(f"Source vocabulary size: {self.src_tokenizer.get_vocab_size()}")
        print(f"Target vocabulary size: {self.tgt_tokenizer.get_vocab_size()}")
    
    def build_model(self, checkpoint):
        """모델 구성 및 가중치 로드 (trainer와 동일한 방식)"""
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
        
        # Gradient Checkpointing 활성화 (trainer와 동일)
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        else:
            print("⚠️  Gradient checkpointing not available")
        
        # 드롭아웃 설정 (trainer와 동일한 방식)
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = model_config['P_drop']
        
        # 모델 가중치 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Model loaded successfully")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size (FP32): {total_params * 4 / (1024**2):.2f} MB")
        print(f"Model size (FP16): {total_params * 2 / (1024**2):.2f} MB")
        
        # 손실 함수 설정 (trainer와 동일한 LabelSmoothingLoss)
        training_config = self.config['training']
        self.criterion = LabelSmoothingLoss(
            self.tgt_tokenizer.get_vocab_size(),
            smoothing=training_config['label_smoothing'],
            ignore_index=0
        )
        
        print(f"✓ Label smoothing loss initialized (smoothing={training_config['label_smoothing']})")
    
    def prepare_data(self, data_type='validation'):
        """평가용 데이터 준비 (trainer와 동일한 방식)"""
        print(f"Preparing {data_type} data with BPE tokenizers...")
        
        # 실제 데이터 로드 (trainer와 동일한 방식)
        train_data, val_data, test_data = load_problem_data(self.config)
        train_src, train_tgt = train_data
        val_src, val_tgt = val_data
        
        print(f"Loaded data:")
        print(f"  - Train pairs: {len(train_src):,}")
        print(f"  - Valid pairs: {len(val_src):,}")
        print(f"  - Test pairs: {len(test_data[0]):,}")
        
        # 데이터 선택
        data_config = self.config['data']
        
        if data_type == 'validation':
            eval_src, eval_tgt = val_src, val_tgt
        elif data_type == 'test':
            eval_src, eval_tgt = test_data
        elif data_type == 'train':
            eval_src, eval_tgt = train_src, train_tgt
        else:
            raise ValueError(f"Unsupported data_type: {data_type}. Use 'train', 'validation', or 'test'")
        
        # 데이터 클리닝 적용 (trainer와 동일한 방식)
        if data_config.get('apply_cleaning', True):
            print("Applying data cleaning...")
            eval_src, eval_tgt = clean_sentence_pairs(eval_src, eval_tgt)
            
            print(f"After cleaning:")
            print(f"  - {data_type.title()} pairs: {len(eval_src):,}")
        
        # 빈 데이터 확인 (trainer와 동일한 방식)
        if not eval_src or not eval_tgt:
            print("⚠️  No evaluation data found! Creating sample data for testing...")
            from src.data_loader import create_data_sample_for_testing
            
            # 샘플 데이터 생성
            create_data_sample_for_testing()
            
            # 다시 로드
            train_data, val_data, test_data = load_problem_data(self.config)
            train_src, train_tgt = train_data
            val_src, val_tgt = val_data
            
            if data_type == 'validation':
                eval_src, eval_tgt = val_src, val_tgt
            elif data_type == 'test':
                eval_src, eval_tgt = test_data
            else:
                eval_src, eval_tgt = train_src, train_tgt
            
            print(f"Using sample data:")
            print(f"  - {data_type.title()} pairs: {len(eval_src):,}")
        
        # BPE 기반 토큰 데이터 로더 생성 (trainer와 동일한 방식)
        batch_tokens = self.config['training']['batch_tokens']
        max_length = data_config['max_length']
        
        print(f"Creating BPE token-based data loader with {batch_tokens} tokens per batch...")
        
        self.eval_loader = create_bpe_token_based_data_loader(
            eval_src, eval_tgt, self.src_tokenizer, self.tgt_tokenizer,
            batch_tokens=batch_tokens, max_length=max_length, shuffle=False
        )
        
        # 배치 정보 출력 (trainer와 동일한 방식)
        sample_batch = next(iter(self.eval_loader))
        src_tokens = (sample_batch['src'] != 0).sum().item()
        tgt_tokens = (sample_batch['tgt_input'] != 0).sum().item()
        total_tokens = src_tokens + tgt_tokens
        
        print(f"Sample batch info:")
        print(f"  - Batch size (sentences): {sample_batch['src'].size(0)}")
        print(f"  - Max sequence length: {sample_batch['src'].size(1)}")
        print(f"  - Source tokens: {src_tokens}")
        print(f"  - Target tokens: {tgt_tokens}")
        print(f"  - Total tokens in batch: {total_tokens}")
        print(f"  - Target batch tokens: {batch_tokens}")
    
    def evaluate_full(self, max_batches=None):
        """전체 데이터에 대한 상세 평가 (trainer의 evaluate와 호환)"""
        print("\nStarting full evaluation with BLEU and Perplexity metrics...")
        
        self.model.eval()
        metrics = EvaluationMetrics()
        
        with torch.no_grad():
            progress_bar = tqdm(self.eval_loader, desc="Evaluating")
            
            for batch_idx, batch in enumerate(progress_bar):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # 데이터 이동 (trainer와 동일한 방식)
                src = batch['src'].to(self.device, non_blocking=True)
                tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
                tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
                
                # 모델 예측 (trainer와 동일한 방식)
                output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
                loss = self.criterion(output, tgt_output)
                predictions = torch.argmax(output, dim=-1)
                
                # NaN/Inf 체크 (trainer와 동일한 안전장치)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  NaN/Inf loss detected in batch {batch_idx}!")
                    print(f"   Loss value: {loss.item()}")
                    print(f"   Output stats: min={output.min():.4f}, max={output.max():.4f}")
                    continue
                
                # 손실 업데이트
                batch_tokens = (tgt_output != 0).sum().item()
                metrics.update_loss(loss.item(), batch_tokens)
                
                # 텍스트로 디코딩하여 BLEU 스코어 계산
                src_texts, tgt_texts, pred_texts = batch_decode_for_evaluation(
                    src, tgt_output, predictions,
                    self.src_tokenizer, self.tgt_tokenizer, pad_token_id=0
                )
                
                # 빈 텍스트 필터링
                valid_pairs = [(pred, tgt) for pred, tgt in zip(pred_texts, tgt_texts) 
                              if pred.strip() and tgt.strip()]
                
                if valid_pairs:
                    valid_preds, valid_tgts = zip(*valid_pairs)
                    metrics.add_predictions(list(valid_preds), list(valid_tgts))
                
                # 진행률 업데이트
                current_summary = metrics.get_summary()
                if current_summary:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{current_summary["average_loss"]:.4f}',
                        'ppl': f'{current_summary["perplexity"]:.1f}',
                        'samples': len(metrics.predictions)
                    })
        
        # 최종 결과 계산 및 출력
        results = metrics.get_summary()
        metrics.print_summary()
        
        return results
    
    def evaluate_samples(self, num_samples=5):
        """몇 개 샘플에 대한 상세 분석 (개별 BLEU 스코어 포함)"""
        print(f"\nEvaluating {num_samples} sample translations...")
        
        self.model.eval()
        samples_evaluated = 0
        sample_metrics = EvaluationMetrics()
        
        with torch.no_grad():
            for batch in self.eval_loader:
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
                predictions = torch.argmax(output, dim=-1)
                
                # 배치 전체를 디코딩
                src_texts, tgt_texts, pred_texts = batch_decode_for_evaluation(
                    src, tgt_output, predictions,
                    self.src_tokenizer, self.tgt_tokenizer, pad_token_id=0
                )
                
                batch_size = src.size(0)
                for i in range(min(batch_size, num_samples - samples_evaluated)):
                    print(f"\n--- Sample {samples_evaluated + 1} ---")
                    print(f"Source: {src_texts[i]}")
                    print(f"Target: {tgt_texts[i]}")
                    print(f"Prediction: {pred_texts[i]}")
                    
                    # 개별 샘플 BLEU 스코어
                    if pred_texts[i].strip() and tgt_texts[i].strip():
                        from sacrebleu import BLEU
                        bleu_scorer = BLEU()
                        sample_bleu = bleu_scorer.sentence_score(pred_texts[i], [tgt_texts[i]])
                        print(f"Sample BLEU: {sample_bleu.score:.2f}")
                    
                    # 토큰 레벨 정확도
                    tgt_seq = tgt_output[i]
                    pred_seq = predictions[i]
                    mask = (tgt_seq != 0)
                    if mask.sum() > 0:
                        accuracy = (tgt_seq[mask] == pred_seq[mask]).float().mean().item()
                        print(f"Token Accuracy: {accuracy:.4f}")
                    
                    # 전체 메트릭에 추가
                    if pred_texts[i].strip() and tgt_texts[i].strip():
                        sample_metrics.add_predictions([pred_texts[i]], [tgt_texts[i]])
                    
                    samples_evaluated += 1
                
                if samples_evaluated >= num_samples:
                    break
        
        print(f"\nSample evaluation completed ({samples_evaluated} samples)")
        
        # 샘플들의 전체 BLEU 스코어
        if len(sample_metrics.predictions) > 0:
            sample_bleu_scores = sample_metrics.compute_bleu()
            print(f"\nOverall Sample BLEU Scores:")
            print(f"BLEU: {sample_bleu_scores['bleu']:.2f}")
            print(f"BLEU-1: {sample_bleu_scores['bleu_1']:.2f}")
            print(f"BLEU-2: {sample_bleu_scores['bleu_2']:.2f}")
            print(f"BLEU-3: {sample_bleu_scores['bleu_3']:.2f}")
            print(f"BLEU-4: {sample_bleu_scores['bleu_4']:.2f}")
    
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
        
        # 평가 결과 시각화
        batch_losses = results.get('batch_losses', [])
        if batch_losses:
            plt.figure(figsize=(15, 5))
            
            # 배치별 손실
            plt.subplot(1, 3, 1)
            plt.plot(batch_losses, alpha=0.7)
            plt.axhline(y=results['average_loss'], color='r', linestyle='--', 
                       label=f'Average: {results["average_loss"]:.4f}')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title('Loss per Batch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 손실 분포
            plt.subplot(1, 3, 2)
            plt.hist(batch_losses, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(x=results['average_loss'], color='r', linestyle='--', 
                       label=f'Average: {results["average_loss"]:.4f}')
            plt.xlabel('Loss')
            plt.ylabel('Frequency')
            plt.title('Loss Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 메트릭 요약
            plt.subplot(1, 3, 3)
            metrics_names = ['Perplexity', 'BLEU', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
            metrics_values = [
                results.get('perplexity', 0),
                results.get('bleu', 0),
                results.get('bleu_1', 0),
                results.get('bleu_2', 0),
                results.get('bleu_3', 0),
                results.get('bleu_4', 0)
            ]
            
            # Perplexity는 스케일이 다르므로 정규화
            normalized_values = metrics_values.copy()
            if normalized_values[0] > 0:  # Perplexity
                normalized_values[0] = min(normalized_values[0] / 10, 100)  # 스케일 조정
            
            bars = plt.bar(metrics_names, normalized_values, alpha=0.7)
            plt.title('Evaluation Metrics')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            
            # 값 표시
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'evaluation_analysis.png'), dpi=300)
            plt.close()
        
        print(f"Results saved to: {output_dir}")
        print(f"  - evaluation_results.json")
        print(f"  - loss_analysis.png")

def main():
    parser = argparse.ArgumentParser(description='체크포인트에서 모델 평가 (trainer 호환)')
    parser.add_argument('checkpoint', type=str, help='체크포인트 파일 경로')
    parser.add_argument('--data_type', type=str, default='validation', 
                       choices=['validation', 'train', 'test'], help='평가할 데이터 타입')
    parser.add_argument('--max_batches', type=int, default=None, 
                       help='최대 평가 배치 수 (None이면 전체, trainer 기본값: 20)')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='상세 분석할 샘플 수')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='결과 저장 디렉토리')
    parser.add_argument('--no_samples', action='store_true', 
                       help='샘플 분석 생략')
    parser.add_argument('--device', type=str, default='auto',
                       help='사용할 디바이스 (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"evaluation_{checkpoint_name}_{timestamp}"
    
    print("Transformer 모델 평가 시작 (Trainer 호환 버전)")
    print("=" * 60)
    print(f"체크포인트: {args.checkpoint}")
    print(f"데이터 타입: {args.data_type}")
    print(f"최대 배치: {args.max_batches or 'All'}")
    print(f"디바이스: {args.device}")
    print(f"출력 디렉토리: {args.output_dir}")
    print("=" * 60)
    
    # 평가자 생성 (trainer와 동일한 device 설정)
    evaluator = ModelEvaluator(args.checkpoint, device=args.device)
    
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
