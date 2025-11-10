"""
Transformer용 Learning Rate Scheduler
"""
import math


class TransformerLRScheduler:
    """Learning rate scheduler for Transformer (Warmup + Decay)
    
    Attention is All You Need 논문의 스케줄링 방식을 따름:
    lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    
    배치 토큰 개수 기준: 25,000 토큰
    """

    def __init__(
        self, 
        optimizer, 
        d_model: int, 
        warmup_steps: int = 4000, 
        batch_tokens: int = 25000,
        base_batch_tokens: int = 25000
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            d_model: 모델 차원 (512, 1024 등)
            warmup_steps: warmup 단계 수 (기본값: 4000)
            batch_tokens: 현재 사용하는 배치당 토큰 수
            base_batch_tokens: 기준 배치당 토큰 수 (25000)
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.batch_tokens = batch_tokens
        self.base_batch_tokens = base_batch_tokens
        
        # 배치 크기 조정 비율 계산
        self.batch_ratio = base_batch_tokens / batch_tokens
        
        print(f"LR Scheduler initialized:")
        print(f"  - d_model: {d_model}")
        print(f"  - warmup_steps: {warmup_steps}")
        print(f"  - batch_tokens: {batch_tokens}")
        print(f"  - base_batch_tokens: {base_batch_tokens}")
        print(f"  - batch_ratio: {self.batch_ratio:.4f}")

    def step(self):
        """Update learning rate after each training step"""
        self.step_num += 1
        lr = self._calculate_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
        return lr

    def _calculate_lr(self):
        """Calculate learning rate based on step number and batch size adjustment
        
        공식: lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
        
        배치 크기 조정:
        - 더 큰 배치 (batch_tokens > 25K): 더 빠른 warmup
        - 더 작은 배치 (batch_tokens < 25K): 더 느린 warmup
        """
        # 배치 크기에 따른 effective step 계산
        effective_step = self.step_num * self.batch_ratio
        effective_warmup = self.warmup_steps * self.batch_ratio
        
        # Transformer 논문의 스케줄링 공식
        d_model_factor = self.d_model ** (-0.5)
        step_factor = min(
            effective_step ** (-0.5),
            effective_step * (effective_warmup ** (-1.5))
        )
        
        lr = d_model_factor * step_factor
        
        return lr
    
    def get_last_lr(self):
        """현재 learning rate 반환 (호환성을 위한 메서드)"""
        return [self._calculate_lr()]
    
    def get_lr_info(self):
        """현재 상태 정보 반환"""
        current_lr = self._calculate_lr()
        effective_step = self.step_num * self.batch_ratio
        effective_warmup = self.warmup_steps * self.batch_ratio
        
        is_warmup = effective_step < effective_warmup
        
        return {
            'step': self.step_num,
            'effective_step': effective_step,
            'effective_warmup': effective_warmup,
            'current_lr': current_lr,
            'is_warmup': is_warmup,
            'warmup_progress': min(effective_step / effective_warmup, 1.0) if effective_warmup > 0 else 1.0
        }


class AdaptiveLRScheduler:
    """배치 크기에 더 적극적으로 적응하는 스케줄러"""
    
    def __init__(
        self, 
        optimizer, 
        d_model: int, 
        warmup_steps: int = 4000,
        batch_tokens: int = 25000,
        min_lr: float = 1e-6,
        max_lr_scale: float = 1.0
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            d_model: 모델 차원
            warmup_steps: warmup 단계 수
            batch_tokens: 현재 배치당 토큰 수
            min_lr: 최소 learning rate
            max_lr_scale: 최대 learning rate 스케일
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.batch_tokens = batch_tokens
        self.min_lr = min_lr
        self.max_lr_scale = max_lr_scale
        self.step_num = 0
        
        # 배치 크기 기반 조정 계수
        self.batch_scale = math.sqrt(batch_tokens / 25000)
        
        print(f"Adaptive LR Scheduler initialized:")
        print(f"  - batch_scale: {self.batch_scale:.4f}")
        print(f"  - min_lr: {min_lr}")
        print(f"  - max_lr_scale: {max_lr_scale}")
    
    def step(self):
        """Update learning rate"""
        self.step_num += 1
        lr = self._calculate_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
        return lr
    
    def _calculate_lr(self):
        """적응적 learning rate 계산"""
        # 기본 Transformer 스케줄링
        d_model_factor = self.d_model ** (-0.5)
        step_factor = min(
            self.step_num ** (-0.5),
            self.step_num * (self.warmup_steps ** (-1.5))
        )
        
        base_lr = d_model_factor * step_factor
        
        # 배치 크기 조정 및 제한 적용
        adjusted_lr = base_lr * self.batch_scale * self.max_lr_scale
        final_lr = max(adjusted_lr, self.min_lr)
        
        return final_lr
    
    def get_last_lr(self):
        """현재 learning rate 반환"""
        return [self._calculate_lr()]
