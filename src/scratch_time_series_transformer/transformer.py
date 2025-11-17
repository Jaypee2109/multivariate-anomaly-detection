# model.py
import torch
import torch.nn as nn
from scratch_time_series_transformer.positional_encoding import PositionalEncoding


def generate_causal_mask(seq_len):
    # Upper triangular matrix with 1 above diagonal
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask  # shape: (seq_len, seq_len)


class TransformerTimeSeries(nn.Module):
    def __init__(
        self,
        input_dim=1,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        output_dim=1,
        forecast_horizon=3,
    ):
        super(TransformerTimeSeries, self).__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.forecast_horizon = forecast_horizon
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, forecast_horizon),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        """
        src shape: (batch_size, seq_len, input_dim)
        """
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        mask = generate_causal_mask(src.size(1)).to(src.device)
        output = self.transformer_encoder(src, mask=mask)
        # out = self.decoder(output[:, -1, :])  # predict next value based on last token
        pooled_output = output.mean(dim=1)  # simple pooling over sequence
        out = self.decoder(pooled_output)  # shape: (batch_size, forecast_horizon)
        return out
