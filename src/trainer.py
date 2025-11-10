"""
Transformer 모델 학습을 위한 공통 클래스들
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from .model import Transformer
from .data_utils import create_tokenizer, create_token_based_data_loader, prepare_sample_data, save_tokenizer


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
        
    def evaluate(self, max_batches=20):
        """평가 (제한된 배치 수로)"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break
                    
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
                loss = self.criterion(output, tgt_output)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self, train_steps=None, save_dir="checkpoints"):
        """전체 학습 프로세스 (스텝 기반)"""
        if train_steps is None:
            train_steps = self.config['training']['train_steps']
        
        os.makedirs(save_dir, exist_ok=True)
        
        training_config = self.config['training']
        eval_every = training_config.get('eval_every', 500)
        save_every = training_config.get('save_every', 1000)
        
        train_losses = []
        val_losses = []
        steps = []
        best_val_loss = float('inf')
        
        print(f"\nStarting training for {train_steps} steps...")
        print(f"Evaluation every {eval_every} steps")
        print(f"Checkpoint save every {save_every} steps")
        print("=" * 60)
        
        start_time = time.time()
        self.model.train()
        
        # 무한 데이터 로더 생성 (train_steps만큼 반복)
        def infinite_dataloader(dataloader):
            while True:
                for batch in dataloader:
                    yield batch
        
        data_iter = infinite_dataloader(self.train_loader)
        running_loss = 0
        log_every = 50  # 50스텝마다 로그 출력
        
        for step in range(1, train_steps + 1):
            batch = next(data_iter)
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
            loss = self.criterion(output, tgt_output)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), training_config['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss += loss.item()
            
            # 주기적 로그 출력
            if step % log_every == 0:
                avg_loss = running_loss / log_every
                current_tokens = (src != 0).sum().item() + (tgt_input != 0).sum().item()
                batch_size = src.size(0)
                
                print(f"Step {step:5d}/{train_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                      f"Batch: {batch_size} sents, {current_tokens} tokens")
                
                train_losses.append(avg_loss)
                steps.append(step)
                running_loss = 0
            
            # 평가
            if step % eval_every == 0:
                print(f"\n--- Evaluation at step {step} ---")
                val_loss = self.evaluate()
                val_losses.append(val_loss)
                
                print(f"Val Loss: {val_loss:.4f}")
                
                # 최고 성능 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'step': step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'config': self.config
                    }, os.path.join(save_dir, 'best_model.pth'))
                    print(f"✓ Best model saved with val loss: {val_loss:.4f}")
                
                self.model.train()  # 평가 후 다시 학습 모드로
                print("-" * 40)
            
            # 체크포인트 저장
            if step % save_every == 0:
                torch.save({
                    'step': step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'config': self.config
                }, os.path.join(save_dir, f'checkpoint_step_{step}.pth'))
                print(f"Checkpoint saved at step {step}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Total steps: {train_steps}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # 학습 곡선 저장
        self.save_training_curves_steps(steps, train_losses, val_losses, save_dir)
        
        # 학습 결과 저장
        results = {
            'config': self.config,
            'train_steps': train_steps,
            'steps': steps,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'training_time': total_time
        }
        
        with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return steps, train_losses, val_losses
    
    def save_training_curves_steps(self, steps, train_losses, val_losses, save_dir):
        """스텝 기반 학습 곡선 저장"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(steps, train_losses, label='Train Loss', color='blue', linewidth=1)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 검증 손실은 eval_every 간격으로만 있으므로 별도 처리
        if val_losses:
            eval_every = self.config['training'].get('eval_every', 500)
            val_steps = list(range(eval_every, len(val_losses) * eval_every + 1, eval_every))
            
            plt.subplot(1, 3, 2)
            plt.plot(val_steps, val_losses, label='Validation Loss', color='red', marker='o', linewidth=2)
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 3)
            # 같은 구간의 train loss와 val loss 비교
            train_at_eval = []
            for val_step in val_steps:
                # 가장 가까운 train step의 loss 찾기
                closest_idx = min(range(len(steps)), key=lambda i: abs(steps[i] - val_step))
                train_at_eval.append(train_losses[closest_idx])
            
            plt.plot(val_steps, train_at_eval, label='Train Loss', color='blue', linewidth=2)
            plt.plot(val_steps, val_losses, label='Validation Loss', color='red', linewidth=2)
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Train vs Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
        plt.close()
