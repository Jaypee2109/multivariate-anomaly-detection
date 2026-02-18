import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class RotaryEmbedding:
    def __init__(self, dim, base=10000):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.registered = False
        self.inv_freq = inv_freq

    def get_embedding(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        return cos, sin

    def apply_rotary(self, x, cos, sin):  # cos, sin: [seq_len, head_dim]
        cos = cos[:, None, None, :]  # -> [seq_len, 1, 1, head_dim]
        sin = sin[:, None, None, :]  # -> [seq_len, 1, 1, head_dim]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)


class OldRotaryEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)  # --- NEW ---
        self.rotary_emb = RotaryEmbedding(d_model // nhead)

    def forward(self, src, src_mask=None, **kwargs):  # existing logic, no need to use is_causal
        seq_len, _, _ = src.size()
        cos, sin = self.rotary_emb.get_embedding(seq_len, src.device)  # flatten seq & batch
        src2d = src.reshape(-1, src.size(-1))  # [seq_len * batch, d_model]
        qkv = F.linear(src2d, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        qkv = qkv.view(src.size(0), src.size(1), -1)  # [seq_len, batch, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)  # q, k shape: [seq_len, batch, d_model]
        seq_len, batch, d_model = q.shape
        head_dim = d_model // self.self_attn.num_heads
        nhead = self.self_attn.num_heads  # reshape to [seq_len, batch, nhead, head_dim]
        q = q.view(seq_len, batch, nhead, head_dim)
        k = k.view(seq_len, batch, nhead, head_dim)  # apply rotary per head
        cos, sin = self.rotary_emb.get_embedding(seq_len, q.device)
        q = self.rotary_emb.apply_rotary(q, cos, sin)
        k = self.rotary_emb.apply_rotary(k, cos, sin)  # reshape back to [seq_len, batch, d_model]
        q = q.view(seq_len, batch, d_model)
        k = k.view(seq_len, batch, d_model)
        attn_output, _ = self.self_attn(q, k, v, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)
