# model.py
import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding


def generate_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


def create_lag_features(x, lag=12):
    """
    x: (batch, seq_len)
    returns: (batch, seq_len - lag + 1, lag)
    """
    # x[:, i:] means "shift x by i"
    # The trick below creates all lag-shifted versions
    lagged = torch.stack(
        [x[:, i : -(lag - i - 1)] if i < lag - 1 else x[:, i:] for i in range(lag)],
        dim=-1,
    )
    return lagged


class TransformerTimeSeries(nn.Module):
    def __init__(
        self,
        input_dim,  # lag + time_features_dim
        model_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        lag=12,
    ):
        super().__init__()
        self.lag = lag
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output: 1 timestep per forward pass
        self.decoder = nn.Linear(model_dim, 1)

    def forward(self, src, time_features=None):
        """
        src:           (batch, seq_len, 1)
        time_features: (batch, seq_len, time_dim) or (seq_len, time_dim)
        """
        batch, seq_len, _ = src.shape
        x_lag = create_lag_features(src[:, :, 0], lag=self.lag)
        seq_len_lag = x_lag.size(1)

        if time_features is not None:
            if time_features.dim() == 2:
                time_features = time_features.unsqueeze(0)
            if time_features.size(0) != x_lag.size(0):
                time_features = time_features.repeat(x_lag.size(0), 1, 1)
            tf = time_features[:, -seq_len_lag:, :]
            x_features = torch.cat([x_lag, tf], dim=-1)
        else:
            x_features = x_lag

        x = self.input_proj(x_features)
        x = self.pos_encoder(x)

        mask = generate_causal_mask(seq_len_lag).to(x.device)
        x = self.transformer_encoder(x, mask=mask)

        out = self.decoder(x[:, -1, :])
        return out
