# model.py
import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding


def generate_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


class TransformerTimeSeries(nn.Module):
    def __init__(
        self,
        input_dim,  # 3 → [value, hour, weekday]
        tfy_dim=2,  # 2 → [hour_next, weekday_next]
        model_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        # Project each input timestep (value + TFx) into model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
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

        # NEW: project TFy into model-dim
        self.tfy_proj = nn.Linear(tfy_dim, model_dim)

        # NEW: combine encoder last-state + tfy projection
        self.decoder = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1),
        )

    def forward(self, src, tfy):
        """
        src : (B, lag, 3)   → values + TFx
        tfy : (B, 2)        → TFy features for prediction timestamp
        """
        B, seq_len, _ = src.shape

        # (B, lag, model_dim)
        x = self.input_proj(src)
        x = self.pos_encoder(x)

        # Causal mask for autoregressive behavior
        mask = generate_causal_mask(seq_len).to(src.device)

        # Transformer encoder output
        x = self.transformer_encoder(x, mask=mask)

        # Take last hidden state (context vector)
        last = x[:, -1, :]  # (B, model_dim)

        # Project TFy
        tfy_proj = self.tfy_proj(tfy)  # (B, model_dim)

        # Concatenate context + TFy
        combined = torch.cat([last, tfy_proj], dim=-1)  # (B, 2*model_dim)

        # Predict next value
        out = self.decoder(combined)  # (B, 1)

        return out
