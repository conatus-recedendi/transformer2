"""
Transformer 모델 학습 스크립트
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import Transformer
from src.data_utils import create_tokenizer, create_data_loader, prepare_sample_data, save_tokenizer

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, ignore_index=0):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size, seq_len, num_classes]
            target: [batch_size, seq_len]
        """
        pred = pred.view(-1, self.num_classes)
        target = target.view(-1)
        
        # 무시할 인덱스 마스크 생성
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        if pred.size(0) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 원-핫 인코딩
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # KL Divergence 계산
        log_pred = torch.log_softmax(pred, dim=1)
        loss = -torch.sum(true_dist * log_pred, dim=1).mean()
        
        return loss

def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx=0):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt_input, src_pad_idx=pad_idx, tgt_pad_idx=pad_idx)
        
        # Loss 계산 (패딩 토큰 무시)
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches

def evaluate(model, dataloader, criterion, device, pad_idx=0):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            output = model(src, tgt_input, src_pad_idx=pad_idx, tgt_pad_idx=pad_idx)
            loss = criterion(output, tgt_output)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir="checkpoints"):
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 옵티마이저와 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = LabelSmoothingLoss(model.output_projection.out_features, smoothing=0.1, ignore_index=0)
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # 검증
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 학습률 스케줄링
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Best model saved with val loss: {val_loss:.4f}")
        
        # 주기적으로 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 학습 곡선 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    plt.show()
    
    return train_losses, val_losses

def main():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 샘플 데이터 준비
    print("Preparing sample data...")
    src_texts, tgt_texts = prepare_sample_data()
    
    # 데이터 분할 (간단한 예시를 위해 같은 데이터 사용)
    train_src = src_texts * 10  # 데이터 확장
    train_tgt = tgt_texts * 10
    val_src = src_texts
    val_tgt = tgt_texts
    
    # 토크나이저 생성
    print("Creating tokenizers...")
    src_tokenizer = create_tokenizer(train_src, vocab_size=5000)
    tgt_tokenizer = create_tokenizer(train_tgt, vocab_size=5000)
    
    # 토크나이저 저장
    os.makedirs("tokenizers", exist_ok=True)
    save_tokenizer(src_tokenizer, "tokenizers/src_tokenizer.json")
    save_tokenizer(tgt_tokenizer, "tokenizers/tgt_tokenizer.json")
    
    # 어휘 크기
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    
    # 데이터 로더 생성
    train_loader = create_data_loader(train_src, train_tgt, src_tokenizer, tgt_tokenizer, 
                                    batch_size=4, max_length=128)
    val_loader = create_data_loader(val_src, val_tgt, src_tokenizer, tgt_tokenizer, 
                                  batch_size=4, max_length=128, shuffle=False)
    
    # 모델 생성
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        max_seq_length=128
    ).to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 학습 시작
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=1e-4,
        device=device
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
