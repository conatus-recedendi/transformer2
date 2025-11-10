"""
Transformer 번역 모델 학습을 위한 메인 스크립트
JSON config 파일을 통한 다양한 아키텍처 변형을 지원합니다.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime
import glob

from src.model import Transformer
from src.data_utils import create_tokenizer, create_token_based_data_loader, prepare_sample_data, save_tokenizer

def load_config(config_path):
    """JSON config 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def get_available_configs():
    """사용 가능한 config 파일 목록 반환"""
    config_files = glob.glob('configs/*.json')
    configs = {}
    for config_file in config_files:
        config_name = os.path.basename(config_file).replace('.json', '')
        try:
            config = load_config(config_file)
            configs[config_name] = {
                'path': config_file,
                'description': config.get('description', 'No description'),
                'config': config
            }
        except Exception as e:
            print(f"Warning: Could not load config {config_file}: {e}")
    return configs

def merge_config_with_args(config, args):
    """커맨드라인 인자로 config 덮어쓰기"""
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_tokens is not None:
        config['training']['batch_tokens'] = args.batch_tokens
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.max_length is not None:
        config['data']['max_length'] = args.max_length
        config['model']['max_seq_length'] = args.max_length
    if args.vocab_size is not None:
        config['data']['vocab_size'] = args.vocab_size
    if args.data_multiplier is not None:
        config['data']['data_multiplier'] = args.data_multiplier
    if args.warmup_steps is not None:
        config['training']['warmup_steps'] = args.warmup_steps
    
    return config

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, ignore_index=0):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.view(-1, self.num_classes)
        target = target.view(-1)
        
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        if pred.size(0) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        log_pred = torch.log_softmax(pred, dim=1)
        loss = -torch.sum(true_dist * log_pred, dim=1).mean()
        
        return loss

class TransformerTrainer:
    def __init__(self, config, device='auto'):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        print(f"Trainer initialized with device: {self.device}")
        print(f"Model config: {config.get('description', 'Custom config')}")
        
    def prepare_data(self):
        """데이터 준비"""
        print("Preparing data...")
        src_texts, tgt_texts = prepare_sample_data()
        
        # config에서 데이터 설정 가져오기
        data_config = self.config['data']
        data_multiplier = data_config['data_multiplier']
        vocab_size = data_config['vocab_size']
        max_length = data_config['max_length']
        
        # 데이터 확장
        train_src = src_texts * data_multiplier
        train_tgt = tgt_texts * data_multiplier
        val_src = src_texts
        val_tgt = tgt_texts
        
        # 토크나이저 생성
        print("Creating tokenizers...")
        self.src_tokenizer = create_tokenizer(train_src, vocab_size=vocab_size)
        self.tgt_tokenizer = create_tokenizer(train_tgt, vocab_size=vocab_size)
        
        # 토크나이저 저장
        os.makedirs("tokenizers", exist_ok=True)
        save_tokenizer(self.src_tokenizer, "tokenizers/src_tokenizer.json")
        save_tokenizer(self.tgt_tokenizer, "tokenizers/tgt_tokenizer.json")
        
        # 토큰 기반 데이터 로더 생성
        batch_tokens = self.config['training']['batch_tokens']
        print(f"Creating token-based data loaders with {batch_tokens} tokens per batch...")
        
        self.train_loader = create_token_based_data_loader(
            train_src, train_tgt, self.src_tokenizer, self.tgt_tokenizer,
            batch_tokens=batch_tokens, max_length=max_length, shuffle=True
        )
        self.val_loader = create_token_based_data_loader(
            val_src, val_tgt, self.src_tokenizer, self.tgt_tokenizer,
            batch_tokens=batch_tokens, max_length=max_length, shuffle=False
        )
        
        print(f"Source vocabulary size: {self.src_tokenizer.get_vocab_size()}")
        print(f"Target vocabulary size: {self.tgt_tokenizer.get_vocab_size()}")
        
        # 배치 정보 출력
        sample_batch = next(iter(self.train_loader))
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
        
    def build_model(self):
        """모델 생성"""
        print("Building model...")
        
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
        
        # 드롭아웃 설정 (모델에 드롭아웃이 있다면)
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = model_config['P_drop']
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")
        
    def setup_training(self):
        """학습 설정"""
        print("Setting up training...")
        
        training_config = self.config['training']
        
        # 옵티마이저
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 손실 함수
        self.criterion = LabelSmoothingLoss(
            self.tgt_tokenizer.get_vocab_size(),
            smoothing=training_config['label_smoothing'],
            ignore_index=0
        )
        
        # 스케줄러 (Warmup)
        warmup_steps = training_config['warmup_steps']
        def lr_lambda(step):
            if step == 0:
                return 0
            return min(step ** (-0.5), step * warmup_steps ** (-1.5))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    def train_epoch(self):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
            loss = self.criterion(output, tgt_output)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 현재 배치의 토큰 수 계산
            current_tokens = (src != 0).sum().item() + (tgt_input != 0).sum().item()
            batch_size = src.size(0)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'batch_size': batch_size,
                'tokens': current_tokens
            })
        
        return total_loss / num_batches
    
    def evaluate_epoch(self):
        """한 에포크 평가"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
                loss = self.criterion(output, tgt_output)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs=None, save_dir="checkpoints"):
        """전체 학습 프로세스"""
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        """전체 학습 프로세스"""
        os.makedirs(save_dir, exist_ok=True)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # 학습
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # 평가
            val_loss = self.evaluate_epoch()
            val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # 최고 성능 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"✓ Best model saved with val loss: {val_loss:.4f}")
            
            # 주기적으로 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        
        # 학습 곡선 저장
        self.save_training_curves(train_losses, val_losses, save_dir)
        
        # 학습 결과 저장
        results = {
            'config': self.config,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'training_time': total_time
        }
        
        with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return train_losses, val_losses
    
    def save_training_curves(self, train_losses, val_losses, save_dir):
        """학습 곡선 저장"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_losses, label='Validation Loss', color='red', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Transformer 번역 모델 학습 (JSON Config 지원)')
    parser.add_argument('--config', type=str, default='base', 
                       help='Config 파일 이름 (configs/ 디렉토리의 .json 파일)')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Config 파일의 전체 경로')
    parser.add_argument('--epochs', type=int, default=None, help='학습 에포크 수 (config 파일 덮어쓰기)')
    parser.add_argument('--batch_tokens', type=int, default=None, help='배치 토큰 수 (config 파일 덮어쓰기)')
    parser.add_argument('--learning_rate', type=float, default=None, help='학습률 (config 파일 덮어쓰기)')
    parser.add_argument('--max_length', type=int, default=None, help='최대 시퀀스 길이 (config 파일 덮어쓰기)')
    parser.add_argument('--vocab_size', type=int, default=None, help='어휘 크기 (config 파일 덮어쓰기)')
    parser.add_argument('--data_multiplier', type=int, default=None, help='데이터 확장 배수 (config 파일 덮어쓰기)')
    parser.add_argument('--warmup_steps', type=int, default=None, help='워밍업 스텝 (config 파일 덮어쓰기)')
    parser.add_argument('--save_dir', type=str, default=None, help='저장 디렉토리')
    parser.add_argument('--list_configs', action='store_true', help='사용 가능한 설정 목록 출력')
    parser.add_argument('--create_config', type=str, default=None, help='새로운 config 파일 생성 (파일명)')
    
    args = parser.parse_args()
    
    # config 파일들이 있는지 확인하고 없으면 생성
    if not os.path.exists('configs'):
        os.makedirs('configs')
        print("configs/ 디렉토리가 생성되었습니다.")
    
    # 사용 가능한 config 목록 출력
    if args.list_configs:
        configs = get_available_configs()
        if not configs:
            print("사용 가능한 config 파일이 없습니다.")
            print("configs/ 디렉토리에 .json 파일을 생성하거나 --create_config 옵션을 사용하세요.")
            return
        
        print("사용 가능한 모델 설정:")
        print("=" * 80)
        for name, info in configs.items():
            config = info['config']
            print(f"{name:15s}: {info['description']}")
            model_cfg = config['model']
            training_cfg = config['training']
            data_cfg = config['data']
            print(f"{'':15s}  Model: N={model_cfg['N']}, d_model={model_cfg['d_model']}, "
                  f"d_ff={model_cfg['d_ff']}, h={model_cfg['h']}")
            print(f"{'':15s}  Training: epochs={training_cfg['epochs']}, "
                  f"batch_tokens={training_cfg['batch_tokens']}, lr={training_cfg['learning_rate']}")
            print(f"{'':15s}  Data: vocab_size={data_cfg['vocab_size']}, "
                  f"max_length={data_cfg['max_length']}")
            print()
        return
    
    # 새로운 config 파일 생성
    if args.create_config:
        create_sample_config(args.create_config)
        return
    
    # Config 파일 로드
    if args.config_path:
        config_path = args.config_path
        config_name = os.path.basename(config_path).replace('.json', '')
    else:
        config_name = args.config
        config_path = f'configs/{config_name}.json'
    
    if not os.path.exists(config_path):
        print(f"Error: Config 파일을 찾을 수 없습니다: {config_path}")
        print("사용 가능한 config 목록을 보려면 --list_configs 옵션을 사용하세요.")
        return
    
    try:
        config = load_config(config_path)
        print(f"Config 파일 로드 완료: {config_path}")
    except Exception as e:
        print(f"Error: Config 파일 로드 실패: {e}")
        return
    
    # 커맨드라인 인자로 config 덮어쓰기
    config = merge_config_with_args(config, args)
    
    # 저장 디렉토리 설정
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"checkpoints_{config_name}_{timestamp}"
    
    print("Transformer 번역 모델 학습 시작")
    print("=" * 80)
    print(f"Config: {config_name} - {config.get('description', 'No description')}")
    print(f"에포크: {config['training']['epochs']}")
    print(f"배치 토큰 수: {config['training']['batch_tokens']}")
    print(f"학습률: {config['training']['learning_rate']}")
    print(f"모델 차원: {config['model']['d_model']}, 레이어: {config['model']['N']}")
    print(f"저장 디렉토리: {args.save_dir}")
    print("=" * 80)
    
    # 트레이너 생성
    trainer = TransformerTrainer(config)
    
    # 데이터 준비
    trainer.prepare_data()
    
    # 모델 구성
    trainer.build_model()
    
    # 학습 설정
    trainer.setup_training()
    
    # 학습 시작
    train_losses, val_losses = trainer.train(
        num_epochs=config['training']['epochs'],
        save_dir=args.save_dir
    )
    
    print(f"\n학습 완료! 결과는 {args.save_dir}에 저장되었습니다.")

def create_sample_config(config_name):
    """샘플 config 파일 생성"""
    sample_config = {
        "model": {
            "N": 6,
            "d_model": 512,
            "d_ff": 2048,
            "h": 8,
            "d_k": 64,
            "d_v": 64,
            "P_drop": 0.1,
            "max_seq_length": 128
        },
        "training": {
            "epochs": 20,
            "batch_tokens": 25000,
            "learning_rate": 1e-4,
            "warmup_steps": 4000,
            "label_smoothing": 0.1,
            "grad_clip": 1.0
        },
        "data": {
            "vocab_size": 5000,
            "max_length": 128,
            "data_multiplier": 10
        },
        "description": f"Custom config: {config_name}"
    }
    
    config_path = f"configs/{config_name}.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    print(f"샘플 config 파일이 생성되었습니다: {config_path}")
    print("파일을 편집하여 원하는 설정으로 변경하세요.")


if __name__ == "__main__":
    main()
