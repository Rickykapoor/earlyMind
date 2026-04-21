"""
src/models/mri_encoder.py
Pretrained EfficientNet-B0 MRI encoder adapted for grayscale infant brain scans.
Input: (B, 3, 1, 64, 64) — 3 slices × grayscale × 64×64
Output: (B, 128)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.config import cfg


class MRIEncoder(nn.Module):
    """
    EfficientNet-B0 backbone modified for single-channel 64×64 MRI slices.
    Three slices are processed independently then fused with cross-attention.

    Architecture:
      - Modified EfficientNet-B0: in_channels 3→1, grayscale-adapted weights
      - Process each of 3 slices: (B, 1, 64, 64) → (B, 1280)
      - Reshape → (B, 3, 1280), apply MultiheadAttention for cross-slice fusion
      - Mean across slices → (B, 1280)
      - MLP projection: 1280 → 256 → 128
    """

    def __init__(
        self,
        embed_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim  = embed_dim

        # ----------------------------------------------------------------
        # Load pretrained EfficientNet-B0 backbone
        # ----------------------------------------------------------------
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Adapt first conv from 3 → 1 channel
        orig_conv: nn.Conv2d = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        # Initialize with mean of RGB weights (preserves learned statistics)
        new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.features[0][0] = new_conv

        # Remove classifier + pooling
        self.backbone = backbone.features   # outputs (B, 1280, 2, 2) for 64×64 input
        self.gap       = nn.AdaptiveAvgPool2d(1)    # → (B, 1280, 1, 1)

        backbone_out = 1280

        # ----------------------------------------------------------------
        # Cross-slice attention fusion
        # ----------------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=backbone_out,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(backbone_out)

        # ----------------------------------------------------------------
        # Projection: 1280 → 256 → 128
        # ----------------------------------------------------------------
        self.proj = nn.Sequential(
            nn.Linear(backbone_out, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
        )

        # ----------------------------------------------------------------
        # Pretraining head (myelination status binary)
        # ----------------------------------------------------------------
        self.pretrain_head = nn.Linear(embed_dim, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 1, 64, 64)
        Returns: (B, 128) embedding
        """
        B = x.shape[0]
        n_slices = x.shape[1]                    # 3

        # Process each slice independently
        # Reshape: (B, 3, 1, H, W) → (B*3, 1, H, W)
        x_flat = x.view(B * n_slices, 1, x.shape[3], x.shape[4])

        feats = self.backbone(x_flat)             # (B*3, 1280, H', W')
        feats = self.gap(feats)                   # (B*3, 1280, 1, 1)
        feats = feats.flatten(1)                  # (B*3, 1280)

        # Reshape → (B, 3, 1280) for cross-slice attention
        feats = feats.view(B, n_slices, -1)       # (B, 3, 1280)

        # Cross-slice multi-head attention
        attn_out, _ = self.cross_attn(feats, feats, feats)   # (B, 3, 1280)
        feats = self.cross_norm(feats + attn_out)             # residual

        # Mean across slices
        pooled = feats.mean(dim=1)                            # (B, 1280)

        return self.proj(pooled)                              # (B, 128)

    def forward(self, x: torch.Tensor, pretrain: bool = False) -> dict:
        """
        x : (B, 3, 1, 64, 64)
        Returns dict with: embedding (B, 128) [, myelin_logit (B, 1)]
        """
        emb = self.forward_features(x)
        out = {"embedding": emb}
        if pretrain:
            out["myelin_logit"] = self.pretrain_head(emb)
        return out


# ---------------------------------------------------------------------------
# MRI Pretraining Dataset
# ---------------------------------------------------------------------------

class MRIPretrainDataset(torch.utils.data.Dataset):
    """
    Dataset for MRI myelination-status pretraining.
    Half real slices (label=0 normal), half augmented-delayed (label=1).
    """

    def __init__(
        self,
        subject_records: list,
        augment: bool = True,
        seed: int = 42,
    ):
        """
        subject_records : list of dicts with keys:
          out_path (str), label (int 0/1)
        """
        import numpy as np
        self.records = subject_records
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self._items = []

        for rec in subject_records:
            data   = np.load(rec["out_path"], allow_pickle=True)
            slices = data["slices"].astype("float32")   # .npz key
            label  = int(data.get("label", rec.get("label", 0)))
            self._items.append((slices, label))

            # Add synthetic delayed-myelination sample for label=0 subjects
            if label == 0:
                from src.data.mri_loader import simulate_delayed_myelination
                delayed = simulate_delayed_myelination(slices, rng=self.rng)
                self._items.append((delayed, 1))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        import numpy as np
        slices, label = self._items[idx]
        if self.augment:
            from src.data.mri_loader import simulate_delayed_myelination
            # Light random augmentation: small probability of applying delay sim
            if self.rng.random() < 0.5:
                severity = self.rng.uniform(0.1, 0.4)
                slices = simulate_delayed_myelination(slices, severity=severity, rng=self.rng)

        # Shape: (3, 64, 64) → (3, 1, 64, 64)
        slices = slices[:, np.newaxis, :, :]
        return (
            torch.from_numpy(slices),
            torch.tensor(label, dtype=torch.float32),
        )
