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
import glob
import re
from collections import defaultdict

from src.model import Transformer
from src.data_utils import create_tokenizer, create_token_based_data_loader, load_tokenizer
from src.trainer import LabelSmoothingLoss
from src.metrics import EvaluationMetrics, batch_decode_for_evaluation
from src.bpe_adapter import load_bpe_tokenizers, create_bpe_token_based_data_loader, save_bpe_tokenizers
from src.data_loader import load_problem_data, clean_sentence_pairs


class BeamSearchDecoder:
    """Beam Search Decoder for Transformer model"""
    
    def __init__(self, model, tgt_tokenizer, beam_size=4, alpha=0.6, max_length_offset=50):
        self.model = model
        self.tgt_tokenizer = tgt_tokenizer
        self.beam_size = beam_size
        self.alpha = alpha  # length penalty
        self.max_length_offset = max_length_offset
        self.pad_token_id = 0
        self.eos_token_id = getattr(tgt_tokenizer, 'eos_token_id', 2)
        self.bos_token_id = getattr(tgt_tokenizer, 'bos_token_id', 1)
        
    def beam_search(self, src, src_mask=None):
        """
        Beam search decoding for a single source sequence
        Args:
            src: [1, src_len] source sequence
            src_mask: [1, src_len] source mask (optional)
        Returns:
            best_sequence: [tgt_len] best decoded sequence
        """
        batch_size = src.size(0)
        assert batch_size == 1, "Beam search currently supports batch_size=1"
        
        device = src.device
        src_len = src.size(1)
        max_length = src_len + self.max_length_offset
        
        # Encode source
        with torch.no_grad():
            # Get encoder output (assuming model has separate encoder method)
            if hasattr(self.model, 'encode'):
                encoder_output = self.model.encode(src, src_mask)
            else:
                # Fallback: run full model with dummy target to get encoder states
                dummy_tgt = torch.tensor([[self.bos_token_id]], device=device)
                _ = self.model(src, dummy_tgt, src_pad_idx=self.pad_token_id, tgt_pad_idx=self.pad_token_id)
                # This is a simplified approach; ideally model should expose encoder
                encoder_output = None
        
        # Initialize beam
        beams = [(torch.tensor([self.bos_token_id], device=device), 0.0)]  # (sequence, score)
        completed_beams = []
        
        for step in range(max_length):
            if len(beams) == 0:
                break
                
            # Collect all current sequences for batch processing
            current_sequences = []
            current_scores = []
            
            for seq, score in beams:
                if seq[-1] == self.eos_token_id:
                    # Apply length penalty and add to completed beams
                    length_penalty = ((5 + len(seq)) / 6) ** self.alpha
                    final_score = score / length_penalty
                    completed_beams.append((seq, final_score))
                else:
                    current_sequences.append(seq)
                    current_scores.append(score)
            
            if not current_sequences:
                break
            
            # Prepare batch input
            max_seq_len = max(len(seq) for seq in current_sequences)
            batch_tgt = torch.full((len(current_sequences), max_seq_len), 
                                 self.pad_token_id, device=device)
            
            for i, seq in enumerate(current_sequences):
                batch_tgt[i, :len(seq)] = seq
            
            # Expand source to match batch size
            batch_src = src.expand(len(current_sequences), -1)
            
            # Get model predictions
            with torch.no_grad():
                output = self.model(batch_src, batch_tgt, 
                                  src_pad_idx=self.pad_token_id, 
                                  tgt_pad_idx=self.pad_token_id)
                
                # Get probabilities for next token (last position)
                next_token_logits = output[:, -1, :]  # [batch_size, vocab_size]
                next_token_probs = torch.log_softmax(next_token_logits, dim=-1)
            
            # Generate new beams
            new_beams = []
            
            for i, (seq, score) in enumerate(zip(current_sequences, current_scores)):
                # Get top-k next tokens
                top_probs, top_indices = torch.topk(next_token_probs[i], self.beam_size)
                
                for prob, token_id in zip(top_probs, top_indices):
                    new_seq = torch.cat([seq, token_id.unsqueeze(0)])
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score))
            
            # Keep only top beam_size beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_size]
        
        # Add remaining beams to completed beams
        for seq, score in beams:
            length_penalty = ((5 + len(seq)) / 6) ** self.alpha
            final_score = score / length_penalty
            completed_beams.append((seq, final_score))
        
        # Return best sequence
        if completed_beams:
            best_seq, best_score = max(completed_beams, key=lambda x: x[1])
            return best_seq[1:]  # Remove BOS token
        else:
            # Fallback to first beam
            return beams[0][0][1:] if beams else torch.tensor([self.eos_token_id], device=device)
    
    def decode_batch(self, src_batch, src_mask_batch=None):
        """
        Decode a batch of sequences using beam search
        Args:
            src_batch: [batch_size, src_len]
            src_mask_batch: [batch_size, src_len] (optional)
        Returns:
            decoded_sequences: list of decoded sequences
        """
        batch_size = src_batch.size(0)
        decoded_sequences = []
        
        for i in range(batch_size):
            src = src_batch[i:i+1]  # [1, src_len]
            src_mask = src_mask_batch[i:i+1] if src_mask_batch is not None else None
            
            decoded_seq = self.beam_search(src, src_mask)
            decoded_sequences.append(decoded_seq)
        
        return decoded_sequences

class ModelEvaluator:
    def __init__(self, checkpoint_path, device='auto', use_averaging=True, use_beam_search=True):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model = None
        self.config = None
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.criterion = None
        self.use_averaging = use_averaging
        self.use_beam_search = use_beam_search
        self.beam_decoder = None
        
        print(f"Evaluator initialized with device: {self.device}")
        print(f"Checkpoint averaging: {'Enabled' if use_averaging else 'Disabled'}")
        print(f"Beam search: {'Enabled' if use_beam_search else 'Disabled'}")
        
    def find_recent_checkpoints(self, checkpoint_dir, max_checkpoints):
        """최근 체크포인트들 찾기"""
        checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_step_*.pth')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return []
        
        # 스텝 번호로 정렬
        def extract_step(filename):
            match = re.search(r'checkpoint_step_(\d+)\.pth', filename)
            return int(match.group(1)) if match else 0
        
        checkpoint_files.sort(key=extract_step, reverse=True)
        return checkpoint_files[:max_checkpoints]
    
    def load_checkpoint(self):
        """체크포인트 로드 (averaging 지원)"""
        if self.use_averaging:
            return self.load_averaged_checkpoint()
        else:
            return self.load_single_checkpoint()
    
    def load_single_checkpoint(self):
        """단일 체크포인트 로드"""
        print(f"Loading single checkpoint from: {self.checkpoint_path}")
        
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
    
    def load_averaged_checkpoint(self):
        """여러 체크포인트 평균하여 로드"""
        print(f"Loading and averaging multiple checkpoints...")
        
        # 체크포인트 디렉토리 찾기
        if os.path.isfile(self.checkpoint_path):
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            
            # config 로드를 위해 첫 번째 체크포인트 로드
            first_checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.config = first_checkpoint['config']
        else:
            checkpoint_dir = self.checkpoint_path
            # 디렉토리에서 가장 최근 체크포인트로 config 로드
            recent_checkpoints = self.find_recent_checkpoints(checkpoint_dir, 1)
            if not recent_checkpoints:
                raise FileNotFoundError(f"No checkpoints found in: {checkpoint_dir}")
            first_checkpoint = torch.load(recent_checkpoints[0], map_location=self.device)
            self.config = first_checkpoint['config']
        
        # config에서 max_checkpoints 읽기 (기본값: 5)
        max_checkpoints = self.config['training'].get('max_checkpoints', 5)
        print(f"  - Using max_checkpoints from config: {max_checkpoints}")
        
        # 최근 체크포인트들 찾기
        recent_checkpoints = self.find_recent_checkpoints(checkpoint_dir, max_checkpoints)
        
        if not recent_checkpoints:
            raise FileNotFoundError(f"No checkpoints found in: {checkpoint_dir}")
        
        print(f"  - Found {len(recent_checkpoints)} checkpoints to average:")
        for i, cp_path in enumerate(recent_checkpoints):
            cp_name = os.path.basename(cp_path)
            print(f"    {i+1}. {cp_name}")
        
        # 체크포인트들 로드 및 평균 계산
        averaged_state_dict = {}
        checkpoint_info = {'steps': [], 'val_losses': []}
        
        for i, cp_path in enumerate(recent_checkpoints):
            print(f"  - Loading checkpoint {i+1}/{len(recent_checkpoints)}: {os.path.basename(cp_path)}")
            checkpoint = torch.load(cp_path, map_location=self.device)
            
            # 정보 수집
            if 'step' in checkpoint:
                checkpoint_info['steps'].append(checkpoint['step'])
            if 'val_loss' in checkpoint:
                checkpoint_info['val_losses'].append(checkpoint['val_loss'])
            
            # 모델 상태 평균화
            model_state = checkpoint['model_state_dict']
            
            if i == 0:
                # 첫 번째 체크포인트로 초기화
                for key, value in model_state.items():
                    averaged_state_dict[key] = value.clone().float()
            else:
                # 평균에 추가
                for key, value in model_state.items():
                    if key in averaged_state_dict:
                        averaged_state_dict[key] += value.float()
        
        # 평균 계산
        num_checkpoints = len(recent_checkpoints)
        for key in averaged_state_dict:
            averaged_state_dict[key] /= num_checkpoints
        
        # 평균화된 체크포인트 생성
        averaged_checkpoint = {
            'model_state_dict': averaged_state_dict,
            'config': self.config,
            'averaged_from': len(recent_checkpoints),
            'checkpoint_steps': checkpoint_info['steps'],
            'checkpoint_val_losses': checkpoint_info['val_losses']
        }
        
        print(f"✓ Averaged {num_checkpoints} checkpoints")
        if checkpoint_info['steps']:
            print(f"  - Step range: {min(checkpoint_info['steps'])} - {max(checkpoint_info['steps'])}")
        if checkpoint_info['val_losses']:
            avg_val_loss = sum(checkpoint_info['val_losses']) / len(checkpoint_info['val_losses'])
            print(f"  - Average validation loss: {avg_val_loss:.4f}")
        
        return averaged_checkpoint
    
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
        
        # Beam Search Decoder 초기화
        if self.use_beam_search:
            # Transformer paper 설정: beam_size=4, alpha=0.6
            beam_size = 4
            alpha = 0.6
            max_length_offset = 50
            
            self.beam_decoder = BeamSearchDecoder(
                model=self.model,
                tgt_tokenizer=self.tgt_tokenizer,
                beam_size=beam_size,
                alpha=alpha,
                max_length_offset=max_length_offset
            )
            print(f"✓ Beam search decoder initialized (beam_size={beam_size}, alpha={alpha}, max_offset={max_length_offset})")
    
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
    
    def evaluate_with_beam_search(self, max_batches=None):
        """Beam Search를 사용한 평가 (Transformer paper 방식)"""
        if not self.use_beam_search or self.beam_decoder is None:
            print("⚠️  Beam search is not enabled. Using greedy decoding instead.")
            return self.evaluate_full(max_batches)
        
        print("\nStarting evaluation with Beam Search (beam_size=4, alpha=0.6)...")
        
        self.model.eval()
        metrics = EvaluationMetrics()
        
        with torch.no_grad():
            progress_bar = tqdm(self.eval_loader, desc="Beam Search Evaluation")
            
            for batch_idx, batch in enumerate(progress_bar):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # 데이터 이동
                src = batch['src'].to(self.device, non_blocking=True)
                tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
                
                # Beam Search 디코딩 (배치별로 처리)
                batch_size = src.size(0)
                beam_predictions = []
                
                for i in range(batch_size):
                    src_seq = src[i:i+1]  # [1, src_len]
                    
                    # Beam search로 디코딩
                    decoded_seq = self.beam_decoder.beam_search(src_seq)
                    beam_predictions.append(decoded_seq)
                
                # 배치 크기에 맞게 패딩
                if beam_predictions:
                    max_pred_len = max(len(pred) for pred in beam_predictions)
                    padded_predictions = torch.full((batch_size, max_pred_len), 
                                                  self.beam_decoder.pad_token_id, 
                                                  device=self.device)
                    
                    for i, pred in enumerate(beam_predictions):
                        padded_predictions[i, :len(pred)] = pred
                    
                    predictions = padded_predictions
                else:
                    continue
                
                # Teacher forcing으로 loss 계산 (beam search와 별개)
                tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
                output = self.model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)
                loss = self.criterion(output, tgt_output)
                
                # NaN/Inf 체크
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  NaN/Inf loss detected in batch {batch_idx}!")
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
                        'bleu': f'{current_summary.get("bleu", 0):.2f}',
                        'samples': len(metrics.predictions)
                    })
        
        # 최종 결과 계산 및 출력
        results = metrics.get_summary()
        print(f"\n🎯 Beam Search Evaluation Results:")
        print(f"   - BLEU Score: {results.get('bleu', 0):.2f}")
        print(f"   - Perplexity: {results.get('perplexity', 0):.2f}")
        print(f"   - Average Loss: {results.get('average_loss', 0):.4f}")
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
    parser = argparse.ArgumentParser(description='체크포인트에서 모델 평가 (Transformer paper 방식)')
    parser.add_argument('checkpoint', type=str, help='체크포인트 파일 또는 디렉토리 경로')
    parser.add_argument('--data_type', type=str, default='validation', 
                       choices=['validation', 'train', 'test'], help='평가할 데이터 타입')
    parser.add_argument('--max_batches', type=int, default=None, 
                       help='최대 평가 배치 수 (None이면 전체)')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='상세 분석할 샘플 수')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='결과 저장 디렉토리')
    parser.add_argument('--no_samples', action='store_true', 
                       help='샘플 분석 생략')
    parser.add_argument('--device', type=str, default='auto',
                       help='사용할 디바이스 (auto/cuda/cpu)')
    parser.add_argument('--no_averaging', action='store_true',
                       help='체크포인트 평균화 비활성화')
    parser.add_argument('--no_beam_search', action='store_true',
                       help='Beam search 비활성화 (greedy decoding 사용)')
    parser.add_argument('--beam_size', type=int, default=4,
                       help='Beam search beam size (기본값: 4)')
    parser.add_argument('--length_penalty', type=float, default=0.6,
                       help='Length penalty alpha (기본값: 0.6)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"evaluation_{checkpoint_name}_{timestamp}"
    
    print("Transformer 모델 평가 시작 (Transformer Paper 방식)")
    print("=" * 70)
    print(f"체크포인트: {args.checkpoint}")
    print(f"데이터 타입: {args.data_type}")
    print(f"최대 배치: {args.max_batches or 'All'}")
    print(f"디바이스: {args.device}")
    print(f"체크포인트 평균화: {'Disabled' if args.no_averaging else 'Enabled'}")
    print(f"Beam Search: {'Disabled' if args.no_beam_search else f'Enabled (size={args.beam_size}, α={args.length_penalty})'}")
    print(f"출력 디렉토리: {args.output_dir}")
    print("=" * 70)
    
    # 평가자 생성
    evaluator = ModelEvaluator(
        args.checkpoint, 
        device=args.device,
        use_averaging=not args.no_averaging,
        use_beam_search=not args.no_beam_search
    )
    
    # 체크포인트 로드
    checkpoint = evaluator.load_checkpoint()
    
    # 토크나이저 로드
    evaluator.load_tokenizers()
    
    # 모델 구성
    evaluator.build_model(checkpoint)
    
    # 데이터 준비
    evaluator.prepare_data(args.data_type)
    
    # Beam search 파라미터 업데이트 (사용자 설정이 있는 경우)
    if evaluator.use_beam_search and evaluator.beam_decoder:
        evaluator.beam_decoder.beam_size = args.beam_size
        evaluator.beam_decoder.alpha = args.length_penalty
        print(f"✓ Beam search parameters updated: beam_size={args.beam_size}, alpha={args.length_penalty}")
    
    # 전체 평가 (Beam search 또는 일반 평가)
    if evaluator.use_beam_search:
        results = evaluator.evaluate_with_beam_search(args.max_batches)
    else:
        results = evaluator.evaluate_full(args.max_batches)
    
    # 샘플 분석
    if not args.no_samples:
        evaluator.evaluate_samples(args.num_samples)
    
    # 결과 저장
    evaluator.save_results(results, args.output_dir)
    
    print(f"\n🎉 평가 완료!")
    print(f"📊 최종 결과:")
    print(f"   - BLEU Score: {results.get('bleu', 0):.2f}")
    print(f"   - Perplexity: {results.get('perplexity', 0):.2f}")
    print(f"   - 평가 방식: {'Beam Search' if evaluator.use_beam_search else 'Greedy'}")
    print(f"   - 체크포인트 평균화: {'Yes' if evaluator.use_averaging else 'No'}")
    print(f"📁 결과 저장 위치: {args.output_dir}")

if __name__ == "__main__":
    main()
