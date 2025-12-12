import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    Time2Vec implementation supporting tensors of shape:
    - (B, in_dim)
    - (B, seq_len, in_dim)
    """

    def __init__(self, in_dim=2, out_dim=16):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # linear: shape (in_dim)
        self.w0 = nn.Parameter(torch.randn(in_dim))
        self.b0 = nn.Parameter(torch.randn(in_dim))

        # periodic: shape (out_dim - 1, in_dim)
        self.W = nn.Parameter(torch.randn(out_dim - 1, in_dim))
        self.B = nn.Parameter(torch.randn(out_dim - 1, in_dim))

    def forward(self, t):
        """
        t: (..., in_dim)
        returns: (..., out_dim)
        """

        # ----- Periodic component -----
        # t:            (..., in_dim)
        # W:            (K, in_dim)
        # result:       (..., K)
        wt = torch.einsum("...i,ki->...k", t, self.W) + torch.einsum(
            "ki,i->k", self.B, torch.ones(self.in_dim, device=t.device)
        )

        # ----- Concatenate -----
        v0 = (t * self.w0).sum(-1, keepdim=True) + self.b0.sum()
        vp = torch.sin(wt)

        return torch.cat([v0, vp], dim=-1)  # (..., 1 + K) = t2v_dim
