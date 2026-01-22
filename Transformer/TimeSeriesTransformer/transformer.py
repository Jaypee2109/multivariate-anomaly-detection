import torch
import torch.nn as nn
from learnableTime2Vec import LearnableTime2VecSinCos
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
        self.input_proj = nn.Linear(1 + (1 + 2 * (t2v_dim - 1)), model_dim)

        # TFy projection
        self.tfy_proj = nn.Linear(1 + 2 * (t2v_dim - 1), model_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, batch_first=True
        )

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

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
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

        mask = generate_causal_mask(seq_len).to(src.device)
        x = self.transformer_encoder(x, mask=mask)

        tfy_embed = self.tfy_t2v(tfy)  # (B, t2v_dim)

        tfy_proj = self.tfy_proj(tfy_embed)  # (B, model_dim)
        tfy_proj = tfy_proj.unsqueeze(1)  # (B, 1, model_dim)

        # Query = TFy, Key/Value = encoder output sequence
        q = tfy_proj
        k = v = x

        attended, _ = self.cross_attention(q, k, v)  # (B, 1, model_dim)

        out = self.decoder(attended.squeeze(1))
        return out


class TransformerTimeSeriesWithLearnableTime2Vec(nn.Module):
    def __init__(
        self,
        t2v_dim=16,
        model_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        dim_feedforward=1024,
        tfx_dim=4,
        tfy_dim=5,
    ):
        super().__init__()

        self.tfx_dim = tfx_dim
        self.tfy_dim = tfy_dim

        # Learnable time2vec embeddings
        self.tfx_t2v = LearnableTime2VecSinCos(in_dim=tfx_dim, out_dim=t2v_dim)
        self.tfy_t2v = LearnableTime2VecSinCos(in_dim=tfy_dim, out_dim=t2v_dim)

        # Input projection: value + time2vec
        self.input_proj = nn.Linear(1 + (1 + 2 * (t2v_dim - 1)), model_dim)

        # TFy projection
        self.tfy_proj = nn.Linear(1 + 2 * (t2v_dim - 1), model_dim)

        # Cross attention with batch_first=True
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, batch_first=True
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Horizon embedding
        self.horizon_embed = nn.Embedding(512, model_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1),
        )

    def forward(self, src, tfy):
        """
        src: (B, seq_len, 1 + tfx_dim)
             → [value, tfx features]
        tfy: (B, tfy_dim)
             → [hour_next, quarter_next, weekday_next, dayofyear_next, series_id, lag_scale]
        """
        B, seq_len, _ = src.shape

        # Split
        vals = src[..., :1]  # (B, seq_len, 1)
        tfx_raw = src[..., 1:]  # (B, seq_len, tfx_dim)

        # Apply LearnableTime2Vec
        tfx_embed = self.tfx_t2v(tfx_raw)  # (B, seq_len, 1 + 2*(t2v_dim-1))

        # Combine value + time2vec
        x = torch.cat([vals, tfx_embed], dim=-1)  # (B, seq_len, input_dim)
        x = self.input_proj(x)  # (B, seq_len, model_dim)

        # Transformer encoder
        mask = generate_causal_mask(seq_len).to(src.device)
        x = self.transformer_encoder(x, mask=mask)  # (B, seq_len, model_dim)

        # TFy embedding for next-step query
        tfy_embed = self.tfy_t2v(tfy)  # (B, 1 + 2*(t2v_dim-1))
        tfy_proj = self.tfy_proj(tfy_embed)  # (B, model_dim)
        tfy_proj = tfy_proj.unsqueeze(1)  # (B, 1, model_dim)

        # Add horizon embedding
        h = torch.zeros(B, dtype=torch.long, device=src.device)  # single step
        tfy_proj = tfy_proj + self.horizon_embed(h).unsqueeze(1)  # (B, 1, model_dim)

        # Cross attention: Query = TFy, Key/Value = encoder output
        attended, _ = self.cross_attention(tfy_proj, x, x)  # (B, 1, model_dim)

        # Decode
        out = self.decoder(attended.squeeze(1))  # (B, 1)
        return out
