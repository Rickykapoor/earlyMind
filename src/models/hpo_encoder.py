"""
src/models/hpo_encoder.py
MLP + self-attention HPO encoder for frequency-weighted phenotype vectors.
Input: (B, n_hpo) — frequency-weighted HPO feature vector
Output: (B, 128)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import cfg


def _next_divisible(n: int, divisor: int) -> int:
    """Return next integer >= n that is divisible by divisor."""
    return n + (divisor - n % divisor) % divisor


class HPOEncoder(nn.Module):
    """
    Self-attention over HPO term features followed by MLP projection.

    Architecture:
      - LayerNorm(n_hpo) input normalization
      - MultiheadAttention self-attention over HPO terms (as sequence of 1 token)
      - Linear projection: n_hpo → 512 → 256 → 128
    Note: n_hpo padded to multiple of 4 for num_heads=4 compatibility.
    """

    def __init__(
        self,
        n_hpo: int = 1024,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_hpo    = n_hpo
        self.embed_dim = embed_dim

        # Pad n_hpo to be divisible by num_heads for MultiheadAttention
        self.n_hpo_padded = _next_divisible(n_hpo, num_heads)

        # ----------------------------------------------------------------
        # Input padding projection (only if needed)
        # ----------------------------------------------------------------
        if self.n_hpo_padded != n_hpo:
            self.input_pad = nn.Linear(n_hpo, self.n_hpo_padded, bias=False)
        else:
            self.input_pad = None

        # ----------------------------------------------------------------
        # Layer norm on input
        # ----------------------------------------------------------------
        self.input_norm = nn.LayerNorm(self.n_hpo_padded)

        # ----------------------------------------------------------------
        # Self-attention: treat the HPO vector as one "token"
        # ----------------------------------------------------------------
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.n_hpo_padded,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.n_hpo_padded)

        # ----------------------------------------------------------------
        # MLP: n_hpo → 512 → 256 → 128
        # ----------------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(self.n_hpo_padded, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )

        # ----------------------------------------------------------------
        # Pretraining head (binary ID classification)
        # ----------------------------------------------------------------
        self.pretrain_head = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, n_hpo)
        Returns: (B, 128) embedding
        """
        # Optional padding
        if self.input_pad is not None:
            x = self.input_pad(x)                # (B, n_hpo_padded)

        x = self.input_norm(x)                   # (B, n_hpo_padded)

        # Self-attention: reshape to (B, 1, n_hpo_padded) as single token
        x_seq = x.unsqueeze(1)                   # (B, 1, n_hpo_padded)
        attn_out, _ = self.self_attn(x_seq, x_seq, x_seq)
        x = self.attn_norm(x + attn_out.squeeze(1))   # residual + (B, n_hpo_padded)

        return self.mlp(x)                        # (B, 128)

    def forward(self, x: torch.Tensor, pretrain: bool = False) -> dict:
        """
        x : (B, n_hpo)
        Returns dict with: embedding (B, 128) [, id_logit (B, 1)]
        """
        emb = self.forward_features(x)
        out = {"embedding": emb}
        if pretrain:
            out["id_logit"] = self.pretrain_head(emb)
        return out


# ---------------------------------------------------------------------------
# HPO Pretraining Dataset
# ---------------------------------------------------------------------------

class HPOPretrainDataset(torch.utils.data.Dataset):
    """
    Dataset for HPO binary ID classification pretraining.
    """

    def __init__(self, X: "np.ndarray", y: "np.ndarray"):
        import numpy as np
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
