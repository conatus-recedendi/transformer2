"""
Transformer 번역 모델 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        batch_size, n_heads, seq_len, d_k = Q.size()
        
        # 🚀 메모리 절약: 긴 시퀀스에서 청크 기반 어텐션 사용
        if seq_len > 512 and self.training:
            return self._memory_efficient_attention(Q, K, V, mask)
        
        # 일반 어텐션 (짧은 시퀀스용)
        # Q, K의 내적을 계산하고 d_k의 제곱근으로 나눔
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 마스크 적용 (패딩이나 미래 토큰 가리기)
        if mask is not None:
            # FP16 호환을 위해 -1e4 사용 (원래 -1e9는 FP16 범위 초과)
            mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, mask_value)
        
        # Softmax 적용
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Value와 가중합
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def _memory_efficient_attention(self, Q, K, V, mask=None, chunk_size=256):
        """🚀 메모리 효율적인 청크 기반 어텐션
        
        핵심 아이디어:
        1. 전체 attention matrix (seq_len x seq_len)를 한번에 만들지 않음
        2. Query를 청크로 나누어 순차적으로 처리
        3. 메모리 사용량: O(seq_len²) → O(chunk_size × seq_len)
        """
        batch_size, n_heads, seq_len, d_k = Q.size()
        scale = 1.0 / math.sqrt(d_k)
        
        # 출력 텐서 미리 할당
        output = torch.zeros_like(Q)
        
        # Query를 청크 단위로 처리
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            Q_chunk = Q[:, :, i:end_i, :]  # [batch, n_heads, chunk_size, d_k]
            
            # 현재 청크에 대해서만 attention scores 계산
            # 메모리: [batch, n_heads, chunk_size, seq_len] (전체보다 훨씬 작음)
            scores_chunk = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale
            
            # 마스크 적용 (해당 청크 부분만)
            if mask is not None:
                mask_chunk = mask[:, :, i:end_i, :]
                mask_value = -1e4 if scores_chunk.dtype == torch.float16 else -1e9
                scores_chunk = scores_chunk.masked_fill(mask_chunk == 0, mask_value)
            
            # Softmax (청크별로 독립적으로 계산)
            attn_weights_chunk = F.softmax(scores_chunk, dim=-1)
            
            # Dropout 적용
            if self.training:
                attn_weights_chunk = self.dropout(attn_weights_chunk)
            
            # Value와 곱셈: [batch, n_heads, chunk_size, d_k]
            output_chunk = torch.matmul(attn_weights_chunk, V)
            
            # 결과를 전체 출력에 저장
            output[:, :, i:end_i, :] = output_chunk
            
            # 🗑️ 중간 텐서 정리 (메모리 절약)
            del scores_chunk, attn_weights_chunk, output_chunk
        
        # attention_weights는 메모리 절약을 위해 None 반환
        return output, None
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformation and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection and layer norm
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_seq_length=5000):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(0.1)
        self.gradient_checkpointing = False  # Gradient checkpointing 플래그

        self.tgt_embedding.weight = self.output_projection.weight
        self.output_projection.weight = self.tgt_embedding.weight
        
    def gradient_checkpointing_enable(self):
        """Gradient checkpointing 활성화"""
        self.gradient_checkpointing = True
        print("✓ Gradient checkpointing enabled - trading compute for memory")
        
    def create_padding_mask(self, seq, pad_idx=0):
        """패딩 토큰에 대한 마스크 생성"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        """미래 토큰을 가리는 마스크 생성"""
        mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        # Create masks
        src_mask = self.create_padding_mask(src, src_pad_idx)
        tgt_mask = self.create_padding_mask(tgt, tgt_pad_idx)
        
        # Look-ahead mask for decoder
        seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(seq_len).to(tgt.device)
        tgt_mask = tgt_mask & look_ahead_mask
        
        # Embedding and positional encoding
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        src_embedded = self.positional_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
        tgt_embedded = self.positional_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        
        src_embedded = self.dropout(src_embedded)
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Encoder with optional gradient checkpointing
        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                encoder_output = checkpoint(encoder_layer, encoder_output, src_mask)
            else:
                encoder_output = encoder_layer(encoder_output, src_mask)
        
        # Decoder with optional gradient checkpointing
        decoder_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            if self.gradient_checkpointing and self.training:
                decoder_output = checkpoint(decoder_layer, decoder_output, encoder_output, src_mask, tgt_mask)
            else:
                decoder_output = decoder_layer(decoder_output, encoder_output, src_mask, tgt_mask)
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output
