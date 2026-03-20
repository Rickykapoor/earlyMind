"""
src/models/eeg_encoder.py
CNN-Transformer EEG encoder for neonatal EEG classification.
Input: (B, 19, 7680) — 19 channels × 30s × 256Hz
Output: (B, 128)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import cfg


class EEGEncoder(nn.Module):
    """
    Multi-scale CNN + Transformer encoder for neonatal EEG signals.

    Architecture:
      - Three parallel Conv1d branches (kernel sizes 3, 25, 125)
      - Concatenate → BatchNorm → AdaptiveAvgPool
      - 2-layer Pre-LN Transformer for temporal context
      - Mean pooling → 2-layer MLP projection to embed_dim=128
    """

    def __init__(
        self,
        in_channels: int = 19,
        embed_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim   = embed_dim

        # ----------------------------------------------------------------
        # Multi-scale parallel conv branches
        # ----------------------------------------------------------------
        branch_out = 64
        self.branch_fast  = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=3,   padding=1,  bias=False),
            nn.GELU(),
        )
        self.branch_theta = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=25,  padding=12, bias=False),
            nn.GELU(),
        )
        self.branch_delta = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=125, padding=62, bias=False),
            nn.GELU(),
        )

        fused_ch = branch_out * 3   # 192

        self.bn = nn.BatchNorm1d(fused_ch)
        self.pool = nn.AdaptiveAvgPool1d(32)  # → (B, 192, 32)

        # ----------------------------------------------------------------
        # Pre-LN TransformerEncoder
        # ----------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_ch,
            nhead=4,
            dim_feedforward=384,
            dropout=0.1,
            batch_first=True,
            norm_first=True,   # Pre-LN for stability on small data
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ----------------------------------------------------------------
        # Projection head: 192 → 256 → 128
        # ----------------------------------------------------------------
        self.proj = nn.Sequential(
            nn.Linear(fused_ch, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
        )

        # ----------------------------------------------------------------
        # Pretraining head (seizure detection binary)
        # ----------------------------------------------------------------
        self.pretrain_head = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 19, T)
        Returns: (B, 128) embedding
        """
        # Multi-scale branches
        f1 = self.branch_fast(x)    # (B, 64, T)
        f2 = self.branch_theta(x)   # (B, 64, T)
        f3 = self.branch_delta(x)   # (B, 64, T)

        h = torch.cat([f1, f2, f3], dim=1)   # (B, 192, T)
        h = self.bn(h)
        h = self.pool(h)                      # (B, 192, 32)
        h = h.permute(0, 2, 1)               # (B, 32, 192)

        h = self.transformer(h)               # (B, 32, 192)
        h = h.mean(dim=1)                     # (B, 192)

        return self.proj(h)                   # (B, 128)

    def forward(self, x: torch.Tensor, pretrain: bool = False) -> dict:
        """
        x : (B, 19, T)
        Returns dict with:
          embedding : (B, 128)
          seizure_logit : (B, 1) — only if pretrain=True
        """
        emb = self.forward_features(x)
        out = {"embedding": emb}
        if pretrain:
            out["seizure_logit"] = self.pretrain_head(emb)
        return out


# ---------------------------------------------------------------------------
# EEG Pretraining Dataset (seizure detection)
# ---------------------------------------------------------------------------

class EEGSeizureDataset(torch.utils.data.Dataset):
    """
    Dataset for EEG seizure-detection pretraining.
    Uses the Helsinki neonatal epoched arrays and binary seizure labels.
    """

    def __init__(
        self,
        subject_records: list,
        augment: bool = False,
        seed: int = 42,
    ):
        """
        subject_records : list of dicts with keys:
          epoch_path (str), label (int 0/1), dq (float)
        """
        import numpy as np
        self.records = subject_records
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        # Pre-build (epoch_array, label) pairs with LOO awareness
        self._items = []
        for rec in subject_records:
            epochs = np.load(rec["epoch_path"], allow_pickle=False)
            lbl    = int(rec.get("label", 0))
            for i in range(len(epochs)):
                self._items.append((epochs[i].astype("float32"), lbl))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        import numpy as np
        epoch, label = self._items[idx]
        if self.augment:
            from src.data.eeg_loader import augment_eeg_epochs
            epoch = augment_eeg_epochs(epoch, rng=self.rng)
        return torch.from_numpy(epoch), torch.tensor(label, dtype=torch.float32)
