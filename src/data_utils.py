"""
데이터 전처리 및 토크나이저 관련 유틸리티
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import json
import os
import random
import numpy as np

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length=512):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        
        # 미리 토크나이징하고 길이 정보 저장
        self.samples = []
        for src_text, tgt_text in zip(src_texts, tgt_texts):
            src_encoding = src_tokenizer.encode(src_text)
            tgt_encoding = tgt_tokenizer.encode(tgt_text)
            
            # 길이 제한
            src_ids = src_encoding.ids[:max_length-1] + [src_tokenizer.token_to_id("[EOS]")]
            tgt_ids = [tgt_tokenizer.token_to_id("[BOS]")] + tgt_encoding.ids[:max_length-2] + [tgt_tokenizer.token_to_id("[EOS]")]
            
            # 실제 길이 (패딩 제외)
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

class TokenBatchSampler:
    """토큰 수 기반 배치 샘플러"""
    def __init__(self, dataset, batch_tokens, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_tokens = batch_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 샘플들을 길이 기준으로 정렬 (비슷한 길이끼리 묶기 위해)
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)
        
    def __iter__(self):
        # 배치 생성
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
            random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        # 대략적인 배치 수 계산
        total_tokens = sum(self.dataset[i]['total_tokens'] for i in range(len(self.dataset)))
        return (total_tokens + self.batch_tokens - 1) // self.batch_tokens

def collate_fn(batch, src_pad_id, tgt_pad_id):
    """토큰 기반 배치를 위한 collate 함수"""
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

def create_tokenizer(texts, vocab_size=10000, model_name="tokenizer"):
    """BPE 토크나이저 생성"""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # 특수 토큰 설정
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        initial_alphabet=[]
    )
    
    # 토크나이저 학습
    if isinstance(texts[0], str):
        tokenizer.train_from_iterator(texts, trainer)
    else:
        tokenizer.train_from_iterator([str(text) for text in texts], trainer)
    
    # 후처리 설정
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]"))
        ]
    )
    
    return tokenizer

def save_tokenizer(tokenizer, path):
    """토크나이저 저장"""
    tokenizer.save(path)

def load_tokenizer(path):
    """토크나이저 로드"""
    return Tokenizer.from_file(path)

def create_data_loader(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, 
                      batch_size=32, max_length=512, shuffle=True):
    """기존 문장 수 기반 데이터 로더 생성 (하위 호환성)"""
    dataset = TranslationDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_token_based_data_loader(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, 
                                 batch_tokens=25000, max_length=512, shuffle=True, drop_last=False):
    """토큰 수 기반 데이터 로더 생성"""
    dataset = TranslationDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length)
    
    # 패딩 토큰 ID
    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")
    
    # 토큰 기반 배치 샘플러
    batch_sampler = TokenBatchSampler(dataset, batch_tokens, shuffle, drop_last)
    
    # collate_fn을 partial로 미리 설정
    from functools import partial
    collate_func = partial(collate_fn, src_pad_id=src_pad_id, tgt_pad_id=tgt_pad_id)
    
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_func)

def load_translation_data(file_path):
    """번역 데이터 로드 (JSON 형식)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    src_texts = [item['src'] for item in data]
    tgt_texts = [item['tgt'] for item in data]
    
    return src_texts, tgt_texts

# prepare_sample_data 함수는 더미 데이터를 위한 것으로 
# 실제 데이터 사용을 위해 src/data_loader.py로 이동되었습니다.
