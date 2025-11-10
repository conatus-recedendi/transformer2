"""
평가 메트릭 계산을 위한 유틸리티
"""
import torch
import numpy as np
from sacrebleu import BLEU
from typing import List, Tuple, Dict, Any


class EvaluationMetrics:
    """평가 메트릭 계산 클래스"""
    
    def __init__(self):
        self.bleu_scorer = BLEU()
        self.reset()
    
    def reset(self):
        """메트릭 초기화"""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.num_batches = 0
        self.predictions = []
        self.references = []
        self.batch_losses = []
        
    def update_loss(self, loss: float, num_tokens: int):
        """손실 업데이트"""
        self.total_loss += loss
        self.total_tokens += num_tokens
        self.num_batches += 1
        self.batch_losses.append(loss)
    
    def add_predictions(self, pred_texts: List[str], ref_texts: List[str]):
        """예측과 참조 텍스트 추가"""
        self.predictions.extend(pred_texts)
        self.references.extend(ref_texts)
    
    def compute_perplexity(self) -> float:
        """퍼플렉서티 계산"""
        if self.num_batches == 0:
            return float('inf')
        
        avg_loss = self.total_loss / self.num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity
    
    def compute_bleu(self) -> Dict[str, float]:
        """BLEU 스코어 계산"""
        if not self.predictions or not self.references:
            return {
                'bleu': 0.0,
                'bleu_1': 0.0,
                'bleu_2': 0.0,
                'bleu_3': 0.0,
                'bleu_4': 0.0
            }
        
        # BLEU 스코어 계산
        bleu_score = self.bleu_scorer.corpus_score(self.predictions, [self.references])
        
        # 개별 n-gram BLEU 스코어
        bleu_1 = BLEU(max_ngram_order=1).corpus_score(self.predictions, [self.references])
        bleu_2 = BLEU(max_ngram_order=2).corpus_score(self.predictions, [self.references])
        bleu_3 = BLEU(max_ngram_order=3).corpus_score(self.predictions, [self.references])
        bleu_4 = BLEU(max_ngram_order=4).corpus_score(self.predictions, [self.references])
        
        return {
            'bleu': bleu_score.score,
            'bleu_1': bleu_1.score,
            'bleu_2': bleu_2.score,
            'bleu_3': bleu_3.score,
            'bleu_4': bleu_4.score
        }
    
    def compute_token_accuracy(self, pred_tokens: torch.Tensor, target_tokens: torch.Tensor, 
                             pad_token_id: int = 0) -> float:
        """토큰 레벨 정확도 계산"""
        mask = (target_tokens != pad_token_id)
        if mask.sum() == 0:
            return 0.0
        
        correct = (pred_tokens[mask] == target_tokens[mask]).sum().item()
        total = mask.sum().item()
        return correct / total
    
    def get_summary(self) -> Dict[str, Any]:
        """전체 평가 결과 요약"""
        if self.num_batches == 0:
            return {}
        
        avg_loss = self.total_loss / self.num_batches
        perplexity = self.compute_perplexity()
        bleu_scores = self.compute_bleu()
        
        # 손실 통계
        batch_losses_array = np.array(self.batch_losses)
        loss_stats = {
            'average_loss': avg_loss,
            'loss_std': np.std(batch_losses_array),
            'loss_min': np.min(batch_losses_array),
            'loss_max': np.max(batch_losses_array)
        }
        
        summary = {
            **loss_stats,
            'perplexity': perplexity,
            **bleu_scores,
            'total_batches': self.num_batches,
            'total_tokens': self.total_tokens,
            'num_samples': len(self.predictions),
            'batch_losses': self.batch_losses
        }
        
        return summary
    
    def print_summary(self):
        """평가 결과 출력"""
        summary = self.get_summary()
        
        if not summary:
            print("No evaluation data available.")
            return
            
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Average Loss: {summary['average_loss']:.4f}")
        print(f"Perplexity: {summary['perplexity']:.2f}")
        print(f"BLEU Score: {summary['bleu']:.2f}")
        print(f"  - BLEU-1: {summary['bleu_1']:.2f}")
        print(f"  - BLEU-2: {summary['bleu_2']:.2f}")
        print(f"  - BLEU-3: {summary['bleu_3']:.2f}")
        print(f"  - BLEU-4: {summary['bleu_4']:.2f}")
        print(f"Total Batches: {summary['total_batches']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"Total Samples: {summary['num_samples']}")
        print(f"Loss Std: {summary['loss_std']:.4f}")
        print(f"Loss Range: [{summary['loss_min']:.4f}, {summary['loss_max']:.4f}]")
        print("="*60)


def decode_tokens_to_text(tokens: torch.Tensor, tokenizer, pad_token_id: int = 0) -> str:
    """토큰을 텍스트로 디코딩"""
    # 패딩 제거
    tokens = tokens[tokens != pad_token_id]
    tokens_list = tokens.cpu().numpy().tolist()
    
    try:
        text = tokenizer.decode(tokens_list)
        # 특수 토큰 제거
        text = text.replace('[BOS]', '').replace('[EOS]', '').replace('[PAD]', '')
        text = text.strip()
        return text
    except Exception as e:
        print(f"Warning: Failed to decode tokens {tokens_list[:10]}...: {e}")
        return ""


def batch_decode_for_evaluation(src_batch: torch.Tensor, 
                              tgt_batch: torch.Tensor, 
                              pred_batch: torch.Tensor,
                              src_tokenizer, 
                              tgt_tokenizer,
                              pad_token_id: int = 0) -> Tuple[List[str], List[str], List[str]]:
    """배치 단위로 디코딩하여 평가용 텍스트 생성"""
    src_texts = []
    tgt_texts = []
    pred_texts = []
    
    batch_size = src_batch.size(0)
    
    for i in range(batch_size):
        # 소스 텍스트
        src_text = decode_tokens_to_text(src_batch[i], src_tokenizer, pad_token_id)
        src_texts.append(src_text)
        
        # 타겟 텍스트
        tgt_text = decode_tokens_to_text(tgt_batch[i], tgt_tokenizer, pad_token_id)
        tgt_texts.append(tgt_text)
        
        # 예측 텍스트
        pred_text = decode_tokens_to_text(pred_batch[i], tgt_tokenizer, pad_token_id)
        pred_texts.append(pred_text)
    
    return src_texts, tgt_texts, pred_texts
