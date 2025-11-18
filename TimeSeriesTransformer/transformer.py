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
        input_dim,  # lag + time feature dim
        model_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

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

        self.decoder = nn.Linear(model_dim, 1)

    def forward(self, src):
        """
        src: (batch, seq_len, input_dim)
        """
        batch, seq_len, _ = src.shape

        x = self.input_proj(src)
        x = self.pos_encoder(x)

        mask = generate_causal_mask(seq_len).to(src.device)

        x = self.transformer_encoder(x, mask=mask)
        out = self.decoder(x[:, -1, :])  # last token → next value

        return out
