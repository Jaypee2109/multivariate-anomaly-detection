import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from time2vec import Time2Vec


def generate_causal_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


class TransformerTimeSeries(nn.Module):
    def __init__(
        self,
        t2v_dim=16,
        model_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        # Time2Vec for TFx (per timestep)
        self.tfx_t2v = Time2Vec(in_dim=2, out_dim=t2v_dim)

        # Time2Vec for TFy (prediction timestamp)
        self.tfy_t2v = Time2Vec(in_dim=2, out_dim=t2v_dim)

        # Input to transformer: [value (1) + time2vec (t2v_dim)]
        self.input_proj = nn.Linear(1 + t2v_dim, model_dim)

        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # TFy projection
        self.tfy_proj = nn.Linear(t2v_dim, model_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1),
        )

    def forward(self, src, tfy):
        """
        src: (B, seq_len, 3) → [value, hour_norm, weekday_norm]
        tfy: (B, 2)          → [hour_norm_next, weekday_norm_next]
        """
        B, seq_len, _ = src.shape

        # Split
        vals = src[..., :1]  # (B, seq_len, 1)
        tfx_raw = src[..., 1:]  # (B, seq_len, 2)

        # Time2Vec for each timestep
        tfx_embed = self.tfx_t2v(tfx_raw)  # (B, seq_len, t2v_dim)

        # Combine
        x = torch.cat([vals, tfx_embed], dim=-1)

        x = self.input_proj(x)
        x = self.pos_encoder(x)

        mask = generate_causal_mask(seq_len).to(src.device)
        x = self.transformer_encoder(x, mask=mask)

        last = x[:, -1, :]

        # TFy time2vec
        tfy_embed = self.tfy_t2v(tfy)  # (B, t2v_dim)
        tfy_proj = self.tfy_proj(tfy_embed)

        out = self.decoder(torch.cat([last, tfy_proj], dim=-1))
        return out
