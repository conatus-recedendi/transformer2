"""
BPE 토크나이저와 기존 시스템 연동을 위한 어댑터
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import logging
from typing import List, Tuple, Dict, Optional
from functools import partial

from .bpe import BPEVocabulary, RealWMTDataset
from .data_utils import prepare_sample_data

logger = logging.getLogger(__name__)


class BPETokenizerAdapter:
    """BPE 토크나이저를 기존 인터페이스에 맞게 변환하는 어댑터"""
    
    def __init__(self, bpe_vocab: BPEVocabulary):
        self.bpe_vocab = bpe_vocab
    
    def get_vocab_size(self) -> int:
        """어휘 크기 반환"""
        return len(self.bpe_vocab)
    
    def encode(self, text: str) -> List[int]:
        """텍스트를 토큰 ID로 인코딩"""
        return self.bpe_vocab.encode(text)
    
    def decode(self, ids: List[int]) -> str:
        """토큰 ID를 텍스트로 디코딩"""
        return self.bpe_vocab.decode(ids)
    
    def token_to_id(self, token: str) -> int:
        """토큰을 ID로 변환"""
        if self.bpe_vocab.sp_model is None:
            raise ValueError("BPE model not loaded")
        return self.bpe_vocab.sp_model.piece_to_id(token)
    
    def id_to_token(self, token_id: int) -> str:
        """ID를 토큰으로 변환"""
        if self.bpe_vocab.sp_model is None:
            raise ValueError("BPE model not loaded")
        return self.bpe_vocab.sp_model.id_to_piece(token_id)


class BPETranslationDataset(Dataset):
    """BPE 기반 번역 데이터셋"""
    
    def __init__(self, src_texts: List[str], tgt_texts: List[str], 
                 src_tokenizer: BPETokenizerAdapter, tgt_tokenizer: BPETokenizerAdapter,
                 max_length: int = 512):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        
        # 샘플 데이터 토크나이징
        self.samples = []
        for src_text, tgt_text in zip(src_texts, tgt_texts):
            src_ids = self.src_tokenizer.encode(src_text)
            tgt_ids = self.tgt_tokenizer.encode(tgt_text)
            
            # 길이 제한 (EOS/BOS 토큰 고려)
            if len(src_ids) > max_length - 1:
                src_ids = src_ids[:max_length-1]
            src_ids = src_ids + [self.src_tokenizer.token_to_id("<EOS>")]
            
            if len(tgt_ids) > max_length - 2:
                tgt_ids = tgt_ids[:max_length-2]
            tgt_ids = [self.tgt_tokenizer.token_to_id("<BOS>")] + tgt_ids + [self.tgt_tokenizer.token_to_id("<EOS>")]
            
            # 실제 길이 계산
            src_len = len(src_ids)
            tgt_len = len(tgt_ids)
            
            self.samples.append({
                'src_ids': src_ids,
                'tgt_ids': tgt_ids,
                'src_len': src_len,
                'tgt_len': tgt_len,
                'total_tokens': src_len + tgt_len
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'src_ids': sample['src_ids'],
            'tgt_ids': sample['tgt_ids'],
            'src_len': sample['src_len'],
            'tgt_len': sample['tgt_len'],
            'total_tokens': sample['total_tokens']
        }


def bpe_collate_fn(batch, src_pad_id, tgt_pad_id):
    """BPE 기반 배치를 위한 collate 함수"""
    src_sequences = []
    tgt_sequences = []
    tgt_input_sequences = []
    tgt_output_sequences = []
    
    for item in batch:
        src_ids = item['src_ids']
        tgt_ids = item['tgt_ids']
        
        src_sequences.append(torch.tensor(src_ids, dtype=torch.long))
        tgt_sequences.append(torch.tensor(tgt_ids, dtype=torch.long))
        tgt_input_sequences.append(torch.tensor(tgt_ids[:-1], dtype=torch.long))  # 마지막 토큰 제외
        tgt_output_sequences.append(torch.tensor(tgt_ids[1:], dtype=torch.long))  # 첫 번째 토큰 제외
    
    # 패딩
    src_padded = pad_sequence(src_sequences, batch_first=True, padding_value=src_pad_id)
    tgt_padded = pad_sequence(tgt_sequences, batch_first=True, padding_value=tgt_pad_id)
    tgt_input_padded = pad_sequence(tgt_input_sequences, batch_first=True, padding_value=tgt_pad_id)
    tgt_output_padded = pad_sequence(tgt_output_sequences, batch_first=True, padding_value=tgt_pad_id)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'tgt_input': tgt_input_padded,
        'tgt_output': tgt_output_padded
    }


class BPETokenBatchSampler:
    """BPE 기반 토큰 배치 샘플러"""
    
    def __init__(self, dataset, batch_tokens, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_tokens = batch_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 샘플들을 인덱스로 관리
        self.indices = list(range(len(dataset)))
        if shuffle:
            import random
            random.shuffle(self.indices)
    
    def __iter__(self):
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx in self.indices:
            sample_tokens = self.dataset[idx]['total_tokens']
            
            # 현재 배치에 추가했을 때 토큰 수가 초과하는지 확인
            if current_tokens + sample_tokens > self.batch_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(idx)
            current_tokens += sample_tokens
        
        # 마지막 배치 처리
        if current_batch and not self.drop_last:
            batches.append(current_batch)
        
        # 배치 순서 섞기
        if self.shuffle:
            import random
            random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        # 대략적인 배치 수 계산
        total_tokens = sum(self.dataset[i]['total_tokens'] for i in range(len(self.dataset)))
        return (total_tokens + self.batch_tokens - 1) // self.batch_tokens


def create_bpe_tokenizers(config: Dict, force_retrain: bool = False) -> Tuple[BPETokenizerAdapter, BPETokenizerAdapter]:
    """BPE 토크나이저 생성 또는 로드"""
    vocab_size = config['data']['vocab_size']
    
    # 모델 파일 경로
    src_model_path = "tokenizers/src_bpe.model"
    tgt_model_path = "tokenizers/tgt_bpe.model"
    
    os.makedirs("tokenizers", exist_ok=True)
    
    # 소스 토크나이저 처리
    src_vocab = BPEVocabulary()
    if os.path.exists(src_model_path) and not force_retrain:
        logger.info(f"Loading existing source BPE model: {src_model_path}")
        src_vocab.load_model(src_model_path)
    else:
        logger.info(f"Training new source BPE model...")
        # 샘플 데이터로 훈련 (실제 사용 시에는 대용량 데이터 사용)
        src_texts, _ = prepare_sample_data()
        
        # 임시 파일로 저장하여 BPE 훈련
        temp_src_file = "temp_src.txt"
        with open(temp_src_file, 'w', encoding='utf-8') as f:
            for text in src_texts * config['data']['data_multiplier']:
                f.write(text + '\n')
        
        src_vocab.train_bpe_model([temp_src_file], vocab_size=vocab_size, model_prefix="tokenizers/src_bpe")
        
        # 임시 파일 삭제
        if os.path.exists(temp_src_file):
            os.remove(temp_src_file)
    
    # 타겟 토크나이저 처리
    tgt_vocab = BPEVocabulary()
    if os.path.exists(tgt_model_path) and not force_retrain:
        logger.info(f"Loading existing target BPE model: {tgt_model_path}")
        tgt_vocab.load_model(tgt_model_path)
    else:
        logger.info(f"Training new target BPE model...")
        # 샘플 데이터로 훈련
        _, tgt_texts = prepare_sample_data()
        
        # 임시 파일로 저장하여 BPE 훈련
        temp_tgt_file = "temp_tgt.txt"
        with open(temp_tgt_file, 'w', encoding='utf-8') as f:
            for text in tgt_texts * config['data']['data_multiplier']:
                f.write(text + '\n')
        
        tgt_vocab.train_bpe_model([temp_tgt_file], vocab_size=vocab_size, model_prefix="tokenizers/tgt_bpe")
        
        # 임시 파일 삭제
        if os.path.exists(temp_tgt_file):
            os.remove(temp_tgt_file)
    
    # 어댑터로 래핑
    src_tokenizer = BPETokenizerAdapter(src_vocab)
    tgt_tokenizer = BPETokenizerAdapter(tgt_vocab)
    
    logger.info(f"BPE tokenizers ready:")
    logger.info(f"  Source vocab size: {src_tokenizer.get_vocab_size()}")
    logger.info(f"  Target vocab size: {tgt_tokenizer.get_vocab_size()}")
    
    return src_tokenizer, tgt_tokenizer


def create_bpe_token_based_data_loader(src_texts: List[str], tgt_texts: List[str], 
                                     src_tokenizer: BPETokenizerAdapter, tgt_tokenizer: BPETokenizerAdapter,
                                     batch_tokens: int = 25000, max_length: int = 512, 
                                     shuffle: bool = True, drop_last: bool = False) -> DataLoader:
    """BPE 기반 토큰 배치 데이터로더 생성"""
    dataset = BPETranslationDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length)
    
    # 패딩 토큰 ID
    src_pad_id = src_tokenizer.token_to_id("<PAD>")
    tgt_pad_id = tgt_tokenizer.token_to_id("<PAD>")
    
    # 토큰 기반 배치 샘플러
    batch_sampler = BPETokenBatchSampler(dataset, batch_tokens, shuffle, drop_last)
    
    # collate_fn 설정
    collate_func = partial(bpe_collate_fn, src_pad_id=src_pad_id, tgt_pad_id=tgt_pad_id)
    
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_func)


def save_bpe_tokenizers(src_tokenizer: BPETokenizerAdapter, tgt_tokenizer: BPETokenizerAdapter):
    """BPE 토크나이저 저장 (이미 모델 파일로 저장되어 있음)"""
    logger.info("BPE tokenizers are automatically saved as .model files")
    logger.info(f"  Source: tokenizers/src_bpe.model")
    logger.info(f"  Target: tokenizers/tgt_bpe.model")


def load_bpe_tokenizers() -> Tuple[BPETokenizerAdapter, BPETokenizerAdapter]:
    """저장된 BPE 토크나이저 로드"""
    src_model_path = "tokenizers/src_bpe.model"
    tgt_model_path = "tokenizers/tgt_bpe.model"
    
    if not os.path.exists(src_model_path) or not os.path.exists(tgt_model_path):
        raise FileNotFoundError(f"BPE model files not found. Expected: {src_model_path}, {tgt_model_path}")
    
    # BPE 어휘 로드
    src_vocab = BPEVocabulary()
    src_vocab.load_model(src_model_path)
    
    tgt_vocab = BPEVocabulary()
    tgt_vocab.load_model(tgt_model_path)
    
    # 어댑터로 래핑
    src_tokenizer = BPETokenizerAdapter(src_vocab)
    tgt_tokenizer = BPETokenizerAdapter(tgt_vocab)
    
    logger.info(f"BPE tokenizers loaded:")
    logger.info(f"  Source vocab size: {src_tokenizer.get_vocab_size()}")
    logger.info(f"  Target vocab size: {tgt_tokenizer.get_vocab_size()}")
    
    return src_tokenizer, tgt_tokenizer
