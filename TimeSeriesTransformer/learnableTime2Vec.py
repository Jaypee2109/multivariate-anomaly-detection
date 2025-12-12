import torch
import torch.nn as nn


class LearnableTime2VecSinCos(nn.Module):
    """
    Fully learnable Time2Vec with sin and cos components
    Supports input shape:
        - (B, in_dim)
        - (B, seq_len, in_dim)
    """

    def __init__(self, in_dim=2, out_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Linear term
        self.w0 = nn.Parameter(torch.randn(in_dim))
        self.b0 = nn.Parameter(torch.randn(1))  # single scalar bias

        # Periodic terms
        self.W = nn.Parameter(torch.randn(out_dim - 1, in_dim))
        self.B = nn.Parameter(torch.randn(out_dim - 1))  # one bias per periodic feature

    def forward(self, t):
        """
        t: (..., in_dim)
        returns: (..., out_dim*2? No, see note)
        """
        # ----- Linear component -----
        v0 = torch.matmul(t, self.w0) + self.b0  # (...,)

        # ----- Periodic component -----
        z = torch.einsum("...i,ki->...k", t, self.W) + self.B  # (..., out_dim-1)
        vp = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)  # (..., 2*(out_dim-1))

        # Concatenate linear + periodic
        return torch.cat([v0.unsqueeze(-1), vp], dim=-1)  # (..., 1 + 2*(out_dim-1))
