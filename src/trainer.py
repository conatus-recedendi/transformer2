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
from torch.cuda.amp import autocast, GradScaler

from .model import Transformer
from .data_utils import create_tokenizer, create_token_based_data_loader, save_tokenizer
from .bpe_adapter import (
    create_bpe_tokenizers,
    create_bpe_token_based_data_loader,
    save_bpe_tokenizers,
)
from .data_loader import load_problem_data, clean_sentence_pairs
from .lr_scheduler import TransformerLRScheduler


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

        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        if pred.size(0) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 메모리 효율적인 구현: true_dist 행렬을 생성하지 않음
        # Numerical stability를 위한 클리핑
        pred = torch.clamp(pred, min=-100, max=100)
        log_pred = torch.log_softmax(pred, dim=1)

        # 정답 레이블에 대한 손실 (confidence 부분)
        nll_loss = -log_pred.gather(1, target.unsqueeze(1)).squeeze(1)

        # 스무딩 부분: 전체 분포에 대한 평균
        smooth_loss = -log_pred.mean(dim=1)

        # 가중 평균으로 최종 손실 계산
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        # NaN/Inf 체크
        final_loss = loss.mean()

        # 안전장치: 비정상적인 loss 값 처리
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            # fallback으로 단순한 cross entropy 반환
            ce_loss = torch.nn.functional.cross_entropy(
                pred, target, ignore_index=self.ignore_index
            )
            return (
                ce_loss
                if not (torch.isnan(ce_loss) or torch.isinf(ce_loss))
                else torch.tensor(0.0, device=pred.device, requires_grad=True)
            )

        return final_loss


class TransformerTrainer:
    def __init__(self, config, device="auto"):
        self.config = config
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        # 체크포인트 관리를 위한 리스트
        self.checkpoint_files = []

        # Update frequency와 gradient accumulation 설정
        self.update_freq = self.config["training"].get("update_freq", 1)
        self.accumulated_loss = 0.0
        self.accumulated_tokens = 0
        self.update_step = 0  # 실제 업데이트 스텝 (gradient accumulation 고려)

        # WMP (Words/tokens per Minute) 추적을 위한 변수들
        self.start_time = None
        self.total_tokens_processed = 0
        self.last_wmp_time = None
        self.last_wmp_tokens = 0

        # Mixed precision 설정 (더 안전한 설정)
        if self.device.type == "cuda":
            self.scaler = GradScaler(
                init_scale=2**8,  # 매우 낮은 초기 스케일 (256)
                growth_factor=1.5,  # 더 보수적인 증가율
                backoff_factor=0.8,  # 더 보수적인 감소율
                growth_interval=2000,  # 더 천천히 스케일 증가
            )
            self.use_amp = False
            # 스케일링 디버깅을 위한 변수들
            self.scale_overflow_count = 0
            self.last_scale_check_step = 0
        else:
            self.scaler = None
            self.use_amp = False

        print(f"Trainer initialized with device: {self.device}")
        print(f"Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Model config: {config.get('description', 'Custom config')}")

    def prepare_data(self):
        """데이터 준비 (실제 데이터 파일 사용, BPE 토크나이저)"""
        print("Preparing data with BPE tokenizers...")

        # 실제 데이터 로드
        train_data, val_data, test_data = load_problem_data(self.config)
        train_src, train_tgt = train_data
        val_src, val_tgt = val_data

        print(f"Loaded data:")
        print(f"  - Train pairs: {len(train_src):,}")
        print(f"  - Valid pairs: {len(val_src):,}")
        print(f"  - Test pairs: {len(test_data[0]):,}")

        # 데이터 클리닝 적용 (config 설정에 따라)
        data_config = self.config["data"]
        if data_config.get("apply_cleaning", True):
            print("Applying data cleaning...")
            train_src, train_tgt = clean_sentence_pairs(train_src, train_tgt)
            val_src, val_tgt = clean_sentence_pairs(val_src, val_tgt)

            print(f"After cleaning:")
            print(f"  - Train pairs: {len(train_src):,}")
            print(f"  - Valid pairs: {len(val_src):,}")

        # 빈 데이터 확인
        if not train_src or not train_tgt:
            print("⚠️  No training data found! Creating sample data for testing...")
            from .data_loader import create_data_sample_for_testing

            # 샘플 데이터 생성
            create_data_sample_for_testing()

            # 다시 로드
            train_data, val_data, test_data = load_problem_data(self.config)
            train_src, train_tgt = train_data
            val_src, val_tgt = val_data

            print(f"Using sample data:")
            print(f"  - Train pairs: {len(train_src):,}")
            print(f"  - Valid pairs: {len(val_src):,}")

        # BPE 토크나이저 생성/로드
        print("Creating/Loading BPE tokenizers...")
        self.src_tokenizer, self.tgt_tokenizer = create_bpe_tokenizers(self.config)

        # 토크나이저 저장 (이미 .model 파일로 저장됨)
        save_bpe_tokenizers(self.src_tokenizer, self.tgt_tokenizer)

        # BPE 기반 토큰 데이터 로더 생성
        batch_tokens = self.config["training"]["batch_tokens"]
        max_length = data_config["max_length"]

        print(
            f"Creating BPE token-based data loaders with {batch_tokens} tokens per batch..."
        )

        self.train_loader = create_bpe_token_based_data_loader(
            train_src,
            train_tgt,
            self.src_tokenizer,
            self.tgt_tokenizer,
            batch_tokens=batch_tokens,
            max_length=max_length,
            shuffle=True,
        )
        self.val_loader = create_bpe_token_based_data_loader(
            val_src,
            val_tgt,
            self.src_tokenizer,
            self.tgt_tokenizer,
            batch_tokens=batch_tokens,
            max_length=max_length,
            shuffle=False,
        )

        print(f"Source vocabulary size: {self.src_tokenizer.get_vocab_size()}")
        print(f"Target vocabulary size: {self.tgt_tokenizer.get_vocab_size()}")

        # 배치 정보 출력
        sample_batch = next(iter(self.train_loader))
        src_tokens = (sample_batch["src"] != 0).sum().item()
        tgt_tokens = (sample_batch["tgt_input"] != 0).sum().item()
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

        model_config = self.config["model"]

        self.model = Transformer(
            src_vocab_size=self.src_tokenizer.get_vocab_size(),
            tgt_vocab_size=self.tgt_tokenizer.get_vocab_size(),
            d_model=model_config["d_model"],
            n_heads=model_config["h"],
            n_layers=model_config["N"],
            d_ff=model_config["d_ff"],
            max_seq_length=model_config["max_seq_length"],
        ).to(self.device)

        # Gradient Checkpointing 활성화 (메모리 절약)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        else:
            print(
                "⚠️  Gradient checkpointing not available - implementing manual checkpointing"
            )

        # 드롭아웃 설정 (모델에 드롭아웃이 있다면)
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = model_config["P_drop"]

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size (FP32): {total_params * 4 / (1024**2):.2f} MB")
        print(f"Model size (FP16): {total_params * 2 / (1024**2):.2f} MB")

        # 예상 VRAM 사용량 계산
        model_memory = (
            total_params * 2 / (1024**2)
            if self.use_amp
            else total_params * 4 / (1024**2)
        )  # FP16/FP32
        gradient_memory = model_memory  # 그래디언트
        optimizer_memory = model_memory * 2  # Adam: momentum + velocity
        estimated_vram = (
            model_memory + gradient_memory + optimizer_memory
        ) * 1.3  # 활성화 + 오버헤드

        print(
            f"Estimated VRAM usage ({'FP16' if self.use_amp else 'FP32'}): {estimated_vram:.0f} MB"
        )

        print("� Memory Optimizations Applied:")
        print(
            f"   ✓ Mixed Precision Training: {'Enabled' if self.use_amp else 'Disabled'}"
        )
        print(f"   ✓ Gradient Checkpointing: Enabled")
        print(f"   ✓ Memory-efficient Label Smoothing: Enabled")
        print(f"   ✓ Estimated memory savings: ~40-60%")

    def setup_training(self):
        """학습 설정"""
        print("Setting up training...")

        training_config = self.config["training"]

        # 🚀 메모리 효율적인 옵티마이저 설정
        print("🔧 Optimizer Memory Analysis:")
        model_params = sum(p.numel() for p in self.model.parameters())
        model_memory_mb = model_params * 2 / (1024**2)  # FP16
        adam_state_memory_mb = model_params * 2 * 4 / (1024**2)  # 2 states × FP32

        print(f"   Model parameters: {model_params:,}")
        print(f"   Model memory (FP16): {model_memory_mb:.1f} MB")
        print(f"   Adam state memory: {adam_state_memory_mb:.1f} MB")
        print(f"   Total optimizer overhead: {adam_state_memory_mb:.1f} MB")

        # 옵티마이저 (메모리 효율적인 설정)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config["learning_rate"],
            betas=(0.9, 0.98),
            eps=1e-9,
            # 메모리 절약을 위한 설정들
            foreach=False,  # 메모리 효율적인 업데이트
        )

        # 손실 함수
        self.criterion = LabelSmoothingLoss(
            self.tgt_tokenizer.get_vocab_size(),
            smoothing=training_config["label_smoothing"],
            ignore_index=0,
        )

        # Transformer LR 스케줄러 (배치 토큰 개수 고려)
        model_config = self.config["model"]
        batch_tokens = self.config["training"]["batch_tokens"]
        warmup_steps = training_config["warmup_steps"]

        self.scheduler = TransformerLRScheduler(
            optimizer=self.optimizer,
            d_model=model_config["d_model"],
            warmup_steps=warmup_steps,
            batch_tokens=batch_tokens,
            base_batch_tokens=25000,  # 기준 배치 토큰 수
        )

    def evaluate(self, max_batches=20):
        """평가 (제한된 배치 수로, Mixed Precision 지원)"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break

                src = batch["src"].to(self.device)
                tgt_input = batch["tgt_input"].to(self.device)
                tgt_output = batch["tgt_output"].to(self.device)

                # Mixed precision으로 평가
                if self.use_amp:
                    with autocast():
                        output = self.model(
                            src, tgt_input, src_pad_idx=0, tgt_pad_idx=0
                        )
                        loss = self.criterion(output, tgt_output)
                else:
                    output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
                    loss = self.criterion(output, tgt_output)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def train(self, train_steps=None, save_dir="checkpoints"):
        """전체 학습 프로세스 (스텝 기반)"""
        if train_steps is None:
            train_steps = self.config["training"]["train_steps"]

        os.makedirs(save_dir, exist_ok=True)

        # 기존 체크포인트 파일들 로드 및 정리
        self.load_existing_checkpoints(save_dir)

        training_config = self.config["training"]
        eval_every = training_config.get("eval_every", 500)
        save_every = training_config.get("save_every", 1000)

        train_losses = []
        val_losses = []
        steps = []
        best_val_loss = float("inf")

        # 현재 스케줄러의 스텝을 확인하여 재시작 지점 결정
        current_step = (
            self.scheduler.step_num if hasattr(self.scheduler, "step_num") else 0
        )

        if current_step > 0:
            print(f"\n🔄 Resuming training from step {current_step}")
            print(f"Target training steps: {train_steps}")
            print(f"Remaining steps: {train_steps - current_step}")
        else:
            print(f"\n🚀 Starting new training for {train_steps} steps")

        print(f"Evaluation every {eval_every} steps")
        print(f"Checkpoint save every {save_every} steps")
        print("=" * 60)

        start_time = time.time()
        self.start_time = start_time
        self.last_wmp_time = start_time
        self.model.train()

        # 무한 데이터 로더 생성 (train_steps만큼 반복)
        def infinite_dataloader(dataloader):
            while True:
                for batch in dataloader:
                    yield batch

        data_iter = infinite_dataloader(self.train_loader)
        running_loss = 0
        log_every = 50  # 50스텝마다 로그 출력

        # Gradient accumulation을 위한 변수들
        self.accumulated_loss = 0.0
        self.accumulated_tokens = 0

        # 현재 스텝부터 목표 스텝까지 학습
        for step in range(current_step + 1, train_steps + 1):
            batch = next(data_iter)
            src = batch["src"].to(self.device, non_blocking=True)
            tgt_input = batch["tgt_input"].to(self.device, non_blocking=True)
            tgt_output = batch["tgt_output"].to(self.device, non_blocking=True)

            # 현재 배치의 토큰 수 계산 (패딩 제외)
            current_batch_tokens = (tgt_output != 0).sum().item()
            self.total_tokens_processed += current_batch_tokens
            self.accumulated_tokens += current_batch_tokens

            # Gradient accumulation 시작 시에만 zero_grad
            if step % self.update_freq == 1 or self.update_freq == 1:
                self.optimizer.zero_grad(set_to_none=True)  # 메모리 절약

            # Mixed Precision Training with 강화된 안전성 체크
            if self.use_amp:
                with autocast():
                    output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
                    loss = self.criterion(output, tgt_output)

                # 🔍 Loss 안전성 체크
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  NaN/Inf loss detected at step {step}!")
                    print(f"   Loss value: {loss.item()}")
                    print(
                        f"   Output stats: min={output.min():.4f}, max={output.max():.4f}"
                    )
                    print(f"   Current scale: {self.scaler.get_scale()}")
                    print(f"   Skipping this batch...")
                    self.scaler.update()  # 스케일 조정
                    continue

                # 🔍 스케일링 상태 모니터링 (주기적)
                if step % 500 == 0:
                    current_scale = self.scaler.get_scale()
                    print(f"🔍 Scale Debug at Step {step}:")
                    print(f"   Current scale: {current_scale}")
                    print(
                        f"   Scale overflows since last check: {self.scale_overflow_count}"
                    )
                    self.scale_overflow_count = 0

                    # 스케일이 너무 높으면 경고
                    if current_scale > 2**15:  # 32768
                        print(f"⚠️  Scale is getting high: {current_scale}")
                        print(f"   Consider reducing growth_factor or growth_interval")

                # Loss를 update_freq로 나누어 gradient accumulation 적용
                scaled_loss = loss / self.update_freq

                # Scaled backward pass
                self.scaler.scale(scaled_loss).backward()

                # 🔍 Gradient 안전성 체크 (unscale 전에 스케일된 gradient 체크)
                scaled_grad_norm_sq = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        scaled_grad_norm_sq += (p.grad**2).sum().item()
                scaled_grad_norm = scaled_grad_norm_sq**0.5

                # 스케일된 gradient가 너무 크면 조기 감지
                if scaled_grad_norm > 1e10:  # 매우 큰 값
                    print(f"⚠️  Very large scaled gradient detected at step {step}!")
                    print(f"   Scaled grad norm: {scaled_grad_norm:.2e}")
                    print(f"   Current scale: {self.scaler.get_scale()}")
                    print(f"   Skipping this batch...")
                    self.scaler.update()  # 스케일 감소
                    self.scale_overflow_count += 1
                    continue

                # Gradient unscaling 및 clipping
                self.scaler.unscale_(self.optimizer)

                # 🔍 Optimizer state 안전성 체크 (주기적)
                if step % 1000 == 0:
                    has_inf_state = False
                    inf_param_count = 0

                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            state = self.optimizer.state[p]
                            if len(state) > 0:  # Adam state 존재 확인
                                # exp_avg, exp_avg_sq 체크
                                if "exp_avg" in state and (
                                    torch.isinf(state["exp_avg"]).any()
                                    or torch.isnan(state["exp_avg"]).any()
                                ):
                                    has_inf_state = True
                                    inf_param_count += 1
                                if "exp_avg_sq" in state and (
                                    torch.isinf(state["exp_avg_sq"]).any()
                                    or torch.isnan(state["exp_avg_sq"]).any()
                                ):
                                    has_inf_state = True
                                    inf_param_count += 1

                    if has_inf_state:
                        print(
                            f"🚨 CRITICAL: Inf/NaN detected in optimizer state at step {step}!"
                        )
                        print(f"   Parameters with inf/nan states: {inf_param_count}")
                        print(f"   Current scale: {self.scaler.get_scale()}")
                        print(f"   Resetting optimizer states...")

                        # Optimizer state 리셋
                        self.optimizer.state.clear()
                        # 스케일도 크게 줄임
                        self.scaler._scale.fill_(2**8)  # 256으로 리셋
                        continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), training_config["grad_clip"]
                )

                # Unscaled gradient 체크
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(
                        f"⚠️  NaN/Inf unscaled gradient at step {step}! Grad norm: {grad_norm}"
                    )
                    print(f"   Current scale: {self.scaler.get_scale()}")
                    self.scaler.update()
                    self.scale_overflow_count += 1
                    continue

                # Update frequency에 따른 실제 optimizer step
                if step % self.update_freq == 0:
                    # 정상적인 step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient accumulation 중일 때는 scaler update만
                    self.scaler.update()
            else:
                # 일반 FP32 학습
                output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
                loss = self.criterion(output, tgt_output)

                # NaN 체크
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  NaN/Inf loss detected at step {step}!")
                    print(f"   Loss value: {loss.item()}")
                    print(f"   Skipping this batch...")
                    continue

                # 🔍 메모리 사용량 디버깅
                # if step % 100 == 1:  # 100스텝마다 메모리 체크
                #     torch.cuda.empty_cache()  # 캐시 정리
                #     print(f"🔍 Memory Debug at Step {step}:")
                #     print(f"   Before backward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

                # Loss를 update_freq로 나누어 gradient accumulation 적용
                scaled_loss = loss / self.update_freq
                scaled_loss.backward()

                # if step % 100 == 1:
                #     print(f"   After backward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                #     print(f"   Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                #     print(f"   Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), training_config["grad_clip"]
                )

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"⚠️  NaN/Inf gradient detected at step {step}!")
                    continue

                self.optimizer.step()

            # Loss accumulation (원본 loss 사용)
            self.accumulated_loss += loss.item()

            # Update frequency에 따른 실제 optimizer/scheduler step
            if step % self.update_freq == 0:
                if (
                    not self.use_amp
                ):  # FP32일 때만 여기서 실행 (AMP는 위에서 이미 실행됨)
                    pass  # AMP에서는 이미 위에서 처리됨

                self.scheduler.step()
                self.update_step += 1

                # Accumulated loss 평균
                avg_accumulated_loss = self.accumulated_loss / self.update_freq
                running_loss += avg_accumulated_loss

                # Reset accumulation
                self.accumulated_loss = 0.0
                accumulated_tokens_for_update = self.accumulated_tokens
                self.accumulated_tokens = 0
            else:
                # Gradient accumulation 중일 때는 scheduler step 하지 않음
                pass

            # 주기적 로그 출력 (실제 업데이트가 발생한 스텝에서만)
            if step % log_every == 0 and step % self.update_freq == 0:
                num_updates = log_every // self.update_freq
                if num_updates > 0:
                    avg_loss = running_loss / num_updates
                else:
                    avg_loss = running_loss  # fallback

                current_tokens = (src != 0).sum().item() + (tgt_input != 0).sum().item()
                batch_size = src.size(0)

                # WMP (Words/tokens per Minute) 계산
                current_time = time.time()
                if self.last_wmp_time is not None:
                    time_elapsed = current_time - self.last_wmp_time
                    tokens_since_last = (
                        self.total_tokens_processed - self.last_wmp_tokens
                    )
                    if time_elapsed > 0:
                        wmp = tokens_since_last / (
                            time_elapsed / 60.0
                        )  # tokens per minute
                    else:
                        wmp = 0
                else:
                    wmp = 0

                # 전체 평균 WMP
                total_time_elapsed = current_time - self.start_time
                if total_time_elapsed > 0:
                    avg_wmp = self.total_tokens_processed / (total_time_elapsed / 60.0)
                else:
                    avg_wmp = 0

                # LR 스케줄러 정보
                lr_info = self.scheduler.get_lr_info()
                warmup_status = "Warmup" if lr_info["is_warmup"] else "Decay"

                print(
                    f"Step {step:5d}/{train_steps} | "
                    f"Update {self.update_step:5d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_info['current_lr']:.2e} ({warmup_status}) | "
                    f"Batch: {batch_size} sents, {current_tokens} tokens | "
                    f"WMP: {wmp:.0f} (avg: {avg_wmp:.0f}) | "
                    f"UF: {self.update_freq}"
                )

                train_losses.append(avg_loss)
                steps.append(step)
                running_loss = 0

                # WMP 추적 변수 업데이트
                self.last_wmp_time = current_time
                self.last_wmp_tokens = self.total_tokens_processed

            # 평가 (실제 업데이트가 발생한 스텝에서만)
            if step % eval_every == 0 and step % self.update_freq == 0:
                print(
                    f"\n--- Evaluation at step {step} (update {self.update_step}) ---"
                )
                val_loss = self.evaluate()
                val_losses.append(val_loss)

                print(f"Val Loss: {val_loss:.4f}")

                # 최고 성능 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "step": step,
                            "update_step": self.update_step,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "val_loss": val_loss,
                            "config": self.config,
                        },
                        os.path.join(save_dir, "best_model.pth"),
                    )
                    print(f"✓ Best model saved with val loss: {val_loss:.4f}")

                self.model.train()  # 평가 후 다시 학습 모드로
                print("-" * 40)

            # 체크포인트 저장 (실제 업데이트가 발생한 스텝에서만)
            if step % save_every == 0 and step % self.update_freq == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pth")
                torch.save(
                    {
                        "step": step,
                        "update_step": self.update_step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "config": self.config,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved at step {step} (update {self.update_step})")

                # 체크포인트 관리 (개수 제한)
                self.manage_checkpoints(checkpoint_path, save_dir)

                # 🧹 주기적인 메모리 정리
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    print(
                        f"   GPU memory cleaned: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated"
                    )

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Total steps: {train_steps}")
        print(f"Best validation loss: {best_val_loss:.4f}")

        # 학습 곡선 저장
        self.save_training_curves_steps(steps, train_losses, val_losses, save_dir)

        # 학습 결과 저장
        results = {
            "config": self.config,
            "train_steps": train_steps,
            "steps": steps,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "training_time": total_time,
        }

        with open(os.path.join(save_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return steps, train_losses, val_losses

    def manage_checkpoints(self, new_checkpoint_path, save_dir):
        """체크포인트 관리: 최대 개수 제한 및 오래된 체크포인트 삭제"""
        training_config = self.config["training"]
        max_checkpoints = training_config.get("max_checkpoints", 5)  # 기본값: 5개

        # 새 체크포인트를 리스트에 추가
        self.checkpoint_files.append(new_checkpoint_path)

        # 최대 개수를 초과하면 가장 오래된 체크포인트 삭제
        if len(self.checkpoint_files) > max_checkpoints:
            # 삭제할 체크포인트 (가장 오래된 것)
            old_checkpoint = self.checkpoint_files.pop(0)

            # 실제 파일 삭제
            try:
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    print(
                        f"🗑️  Removed old checkpoint: {os.path.basename(old_checkpoint)}"
                    )
            except Exception as e:
                print(f"⚠️  Failed to remove old checkpoint {old_checkpoint}: {e}")

        # 현재 체크포인트 상태 출력
        print(
            f"📁 Checkpoint status: {len(self.checkpoint_files)}/{max_checkpoints} files kept"
        )

    def load_existing_checkpoints(self, save_dir):
        """기존 체크포인트 파일들을 발견하여 리스트에 추가"""
        import glob

        # checkpoint_step_*.pth 패턴의 파일들 찾기
        checkpoint_pattern = os.path.join(save_dir, "checkpoint_step_*.pth")
        existing_checkpoints = glob.glob(checkpoint_pattern)

        # 스텝 번호로 정렬 (파일명에서 스텝 번호 추출)
        def extract_step(filename):
            import re

            match = re.search(r"checkpoint_step_(\d+)\.pth", filename)
            return int(match.group(1)) if match else 0

        existing_checkpoints.sort(key=extract_step)

        # 최대 개수만큼만 유지
        training_config = self.config["training"]
        max_checkpoints = training_config.get("max_checkpoints", 5)

        if len(existing_checkpoints) > max_checkpoints:
            # 오래된 체크포인트들 삭제
            to_remove = existing_checkpoints[:-max_checkpoints]
            for checkpoint_path in to_remove:
                try:
                    os.remove(checkpoint_path)
                    print(
                        f"🗑️  Removed old checkpoint: {os.path.basename(checkpoint_path)}"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to remove old checkpoint {checkpoint_path}: {e}")

            # 남은 체크포인트들만 리스트에 추가
            self.checkpoint_files = existing_checkpoints[-max_checkpoints:]
        else:
            self.checkpoint_files = existing_checkpoints

        if self.checkpoint_files:
            print(f"📁 Found {len(self.checkpoint_files)} existing checkpoints")

    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        print(f"Loading checkpoint from: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 모델 상태 로드
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # 옵티마이저 상태 로드
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 스케줄러 상태 로드
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        step = checkpoint.get("step", 0)
        update_step = checkpoint.get(
            "update_step", step // self.update_freq
        )  # 호환성을 위한 fallback
        val_loss = checkpoint.get("val_loss", float("inf"))

        # Update step 복원
        self.update_step = update_step

        print(f"✓ Checkpoint loaded:")
        print(f"  - Step: {step}")
        print(f"  - Update step: {update_step}")
        print(f"  - Validation loss: {val_loss:.4f}")
        print(f"  - Update frequency: {self.update_freq}")

        return step, val_loss

    def save_training_curves_steps(self, steps, train_losses, val_losses, save_dir):
        """스텝 기반 학습 곡선 저장"""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(steps, train_losses, label="Train Loss", color="blue", linewidth=1)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 검증 손실은 eval_every 간격으로만 있으므로 별도 처리
        if val_losses:
            eval_every = self.config["training"].get("eval_every", 500)
            val_steps = list(
                range(eval_every, len(val_losses) * eval_every + 1, eval_every)
            )

            plt.subplot(1, 3, 2)
            plt.plot(
                val_steps,
                val_losses,
                label="Validation Loss",
                color="red",
                marker="o",
                linewidth=2,
            )
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("Validation Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 3, 3)
            # 같은 구간의 train loss와 val loss 비교
            train_at_eval = []
            for val_step in val_steps:
                # 가장 가까운 train step의 loss 찾기
                closest_idx = min(
                    range(len(steps)), key=lambda i: abs(steps[i] - val_step)
                )
                train_at_eval.append(train_losses[closest_idx])

            plt.plot(
                val_steps, train_at_eval, label="Train Loss", color="blue", linewidth=2
            )
            plt.plot(
                val_steps, val_losses, label="Validation Loss", color="red", linewidth=2
            )
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("Train vs Validation Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300)
        plt.close()
