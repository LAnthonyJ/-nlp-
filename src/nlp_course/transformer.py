# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
import copy

class TokenEmbedding(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create a position matrix (max_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(pos * div_term)   # sine for even dimensions
        pe[:, 1::2] = torch.cos(pos * div_term[:embed_dim // 2])   # cosine for odd dimensions
        pe = pe.unsqueeze(0)                      # add batch dimension (1,max_len,embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence exceeds configured positional capacity")
        # Add positional encoding to input x, then apply dropout
        x = x + self.pe[:, :x.size(1), :].detach()  # 位置编码不参与梯度更新
        return self.dropout(x)


class LayerNorm(nn.Module):
    """Feature-wise affine normalization using population variance."""
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        return torch.nn.functional.layer_norm(
            x, self.gamma.shape, self.gamma, self.beta, self.eps)


class FeedForward(nn.Module):
    """Two-layer feed-forward network with ReLU and Dropout"""
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Two linear layers with ReLU and Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if num_heads <= 0 or embed_dim <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        if any(x.ndim != 3 for x in (q, k, v)):
            raise ValueError("q, k and v must have shape [batch, sequence, features]")
        if not q.size(0) == k.size(0) == v.size(0):
            raise ValueError("q, k and v must have the same batch size")
        if k.size(1) != v.size(1):
            raise ValueError("k and v must have the same sequence length")
        if any(x.size(2) != self.q_linear.in_features for x in (q, k, v)):
            raise ValueError("q, k and v feature dimensions must match embed_dim")
        B = q.size(0)

        Q = self.q_linear(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, heads, seq_len, seq_len)

        allowed = None
        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("Mask must be boolean; True means attention is allowed")
            # 2D means padding [B,K]; causal masks must be [1,Q,K] or [B,Q,K].
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            elif mask.ndim != 4:
                raise ValueError("Expected mask [B,K], [B,Q,K] or [B,H,Q,K]")
            try:
                allowed = torch.broadcast_to(mask.to(q.device), attn_scores.shape)
            except RuntimeError as exc:
                raise ValueError("Mask is not compatible with attention dimensions") from exc
            attn_scores = attn_scores.masked_fill(~allowed, float("-inf"))
            # Fully masked rows receive zero attention, without NaN gradients.
            attn_scores = attn_scores.masked_fill(~allowed.any(dim=-1, keepdim=True), 0)
        attn = torch.softmax(attn_scores, dim=-1)
        if allowed is not None:
            attn = attn.masked_fill(~allowed, 0)
        attn = self.drop(attn)

        # 输出
        out = attn @ V  # (B, heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)  # 拼接多头
        return self.out_linear(out)


class Residual(nn.Module):
    """Residual connection with LayerNorm"""
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Pre-normalization residual block.
        return x + self.drop(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Single encoder layer: self-attention + feed-forward"""
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)
        self.res_layers = nn.ModuleList([Residual(embed_dim, dropout) for _ in range(2)])

    def forward(self, x, mask):
        # Self-attention + residual
        x = self.res_layers[0](x, lambda x: self.self_attn(x, x, x, mask))
        # Feed-forward + residual
        x = self.res_layers[1](x, self.ffn)
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        if N < 1:
            raise ValueError("N must be positive")
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.self_attn.num_heads * layer.self_attn.d_k)

    def forward(self, x, mask):
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Single decoder layer: self-attention, cross-attention, feed-forward"""
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)
        self.res_layers = nn.ModuleList([Residual(embed_dim, dropout) for _ in range(3)])

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.res_layers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.res_layers[1](x, lambda x: self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.res_layers[2](x, self.ffn)
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        if N < 1:
            raise ValueError("N must be positive")
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.self_attn.num_heads * layer.self_attn.d_k)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        for l in self.layers:
            x = l(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


class OutputLayer(nn.Module):
    """Projection layer from hidden dimension to vocabulary size"""
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """Full Transformer model"""
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: TokenEmbedding, tgt_embed: TokenEmbedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 out_layer: OutputLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.out_layer = out_layer

    def encode(self, src, src_mask):
        src = self.src_pos(self.src_embed(src))
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        tgt = self.tgt_pos(self.tgt_embed(tgt))
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)

    def project(self, x):
        return self.out_layer(x)


def build_transformer(src_vocab: int, tgt_vocab: int, src_len: int, tgt_len: int,
                     embed_dim: int = 256, N: int = 4, heads: int = 8,
                     hidden_dim: int = 512, dropout: float = 0.1):
    """Helper function to build a Transformer model with given parameters"""
    src_embed = TokenEmbedding(embed_dim, src_vocab)
    tgt_embed = TokenEmbedding(embed_dim, tgt_vocab)
    src_pos = PositionalEncoding(embed_dim, src_len, dropout)
    tgt_pos = PositionalEncoding(embed_dim, tgt_len, dropout)

    enc_layer = EncoderLayer(embed_dim, heads, hidden_dim, dropout)
    dec_layer = DecoderLayer(embed_dim, heads, hidden_dim, dropout)

    encoder = Encoder(enc_layer, N)
    decoder = Decoder(dec_layer, N)

    out_layer = OutputLayer(embed_dim, tgt_vocab)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, out_layer)

    # Parameter initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def causal_mask(length: int, device=None):
    """Return boolean [1,Q,K]; True allows attending to current/past tokens."""
    if length < 1:
        raise ValueError("length must be positive")
    return torch.ones(length, length, dtype=torch.bool, device=device).tril().unsqueeze(0)
