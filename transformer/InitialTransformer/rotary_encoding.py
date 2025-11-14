import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_embedding(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        return cos, sin

    def apply_rotary(self, x, cos, sin):
        cos = cos[:, None, None, :]  # [seq_len, 1, 1, head_dim]
        sin = sin[:, None, None, :]  # [seq_len, 1, 1, head_dim]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)


class RotaryEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(d_model // nhead)
        self.nhead = nhead
        self.d_model = d_model

    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        seq_len, batch, _ = src.size()
        cos, sin = self.rotary_emb.get_embedding(seq_len, src.device)

        # Compute QKV manually (3 * d_model)
        qkv = F.linear(src, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split into heads: [seq_len, batch, nhead, head_dim]
        head_dim = self.d_model // self.self_attn.num_heads
        nhead = self.self_attn.num_heads
        q = q.view(seq_len, batch, nhead, head_dim)
        k = k.view(seq_len, batch, nhead, head_dim)
        v = v.view(seq_len, batch, nhead, head_dim)

        # Apply rotary to q and k *per head*
        q = self.rotary_emb.apply_rotary(q, cos, sin)
        k = self.rotary_emb.apply_rotary(k, cos, sin)

        # Merge heads back: [seq_len, batch, d_model]
        q = q.view(seq_len, batch, self.d_model)
        k = k.view(seq_len, batch, self.d_model)
        v = v.view(seq_len, batch, self.d_model)

        # Run attention
        attn_output, _ = self.self_attn(q, k, v, attn_mask=src_mask)

        # Add & norm
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)
