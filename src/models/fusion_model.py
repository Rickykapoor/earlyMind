"""
src/models/fusion_model.py
Late-Fusion Transformer that combines EEG, MRI, and HPO encoder embeddings.
Supports any combination of available modalities at inference time.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import cfg
from src.models.eeg_encoder import EEGEncoder
from src.models.mri_encoder import MRIEncoder
from src.models.hpo_encoder import HPOEncoder


# Modality index for type embeddings
MODALITY_IDX = {"cls": 0, "eeg": 1, "mri": 2, "hpo": 3}


class LateFusionTransformer(nn.Module):
    """
    Late-fusion Transformer that takes frozen/fine-tuned modality embeddings
    and learns a joint representation via CLS-token pooling.

    Supported modalities: EEG, MRI, HPO (any subset at inference).
    Missing modalities are handled via zero-masking + key_padding_mask.

    Outputs:
      logits           : (B, 2) — classification
      severity         : (B, 1) — DQ estimate ∈ [0, 100]
      modality_importance : (3,) — softmax weights over [EEG, MRI, HPO]
      cls_embedding    : (B, 128)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        n_hpo: int = 1024,
        freeze_encoders: bool = False,
        eeg_ckpt: Optional[str] = None,
        mri_ckpt: Optional[str] = None,
        hpo_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ----------------------------------------------------------------
        # Sub-encoders
        # ----------------------------------------------------------------
        self.eeg_encoder = EEGEncoder(embed_dim=embed_dim)
        self.mri_encoder = MRIEncoder(embed_dim=embed_dim)
        self.hpo_encoder = HPOEncoder(n_hpo=n_hpo, embed_dim=embed_dim)

        # Optionally load pretrained checkpoints
        if eeg_ckpt is not None:
            self.eeg_encoder.load_state_dict(
                torch.load(eeg_ckpt, map_location="cpu"), strict=False
            )
        if mri_ckpt is not None:
            self.mri_encoder.load_state_dict(
                torch.load(mri_ckpt, map_location="cpu"), strict=False
            )
        if hpo_ckpt is not None:
            self.hpo_encoder.load_state_dict(
                torch.load(hpo_ckpt, map_location="cpu"), strict=False
            )

        if freeze_encoders:
            self._freeze_encoders()

        # ----------------------------------------------------------------
        # CLS token (learnable)
        # ----------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # ----------------------------------------------------------------
        # Modality type embeddings: 4 indices (cls=0, eeg=1, mri=2, hpo=3)
        # ----------------------------------------------------------------
        self.type_embedding = nn.Embedding(4, embed_dim)

        # ----------------------------------------------------------------
        # Fusion Transformer
        # ----------------------------------------------------------------
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_layer, num_layers=n_layers
        )

        # ----------------------------------------------------------------
        # Learnable modality importance (interpretability)
        # ----------------------------------------------------------------
        self.modality_importance = nn.Parameter(
            torch.ones(3) / 3.0   # [eeg_w, mri_w, hpo_w]
        )

        # ----------------------------------------------------------------
        # Classification head: (B, 128) → (B, 2)
        # ----------------------------------------------------------------
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        # ----------------------------------------------------------------
        # Severity (DQ) regression head: (B, 128) → (B, 1) ∈ [0, 100]
        # ----------------------------------------------------------------
        self.severity_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),    # → [0, 1] × 100 = DQ
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _freeze_encoders(self):
        for enc in [self.eeg_encoder, self.mri_encoder, self.hpo_encoder]:
            for param in enc.parameters():
                param.requires_grad = False

    def unfreeze_encoders(self):
        for enc in [self.eeg_encoder, self.mri_encoder, self.hpo_encoder]:
            for param in enc.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        missing_modalities: Optional[List[List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        batch : dict with optional keys 'eeg', 'mri', 'hpo'
          eeg : (B, 19, 7680)
          mri : (B, 3, 1, 64, 64) OR (B, 3, 64, 64) — auto-unsqueezed
          hpo : (B, n_hpo)
        missing_modalities : list of lists, one per sample
          e.g. [["mri"], ["eeg", "hpo"]] — modalities absent for each sample

        Returns
        -------
        dict with: logits, severity, modality_importance, cls_embedding
        """
        device = self.cls_token.device
        B = _infer_batch_size(batch)

        # ----------------------------------------------------------------
        # Encode available modalities
        # ----------------------------------------------------------------
        eeg_emb = mri_emb = hpo_emb = None

        if "eeg" in batch and batch["eeg"] is not None:
            eeg_emb = self.eeg_encoder.forward_features(batch["eeg"])   # (B, 128)
        if "mri" in batch and batch["mri"] is not None:
            mri_in = batch["mri"]
            if mri_in.ndim == 4:   # (B, 3, H, W) → (B, 3, 1, H, W)
                mri_in = mri_in.unsqueeze(2)
            mri_emb = self.mri_encoder.forward_features(mri_in)         # (B, 128)
        if "hpo" in batch and batch["hpo"] is not None:
            hpo_emb = self.hpo_encoder.forward_features(batch["hpo"])   # (B, 128)

        # ----------------------------------------------------------------
        # Build token sequence: [CLS, EEG?, MRI?, HPO?]
        # ----------------------------------------------------------------
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, 128)
        cls = cls + self.type_embedding(
            torch.tensor([MODALITY_IDX["cls"]], device=device)
        )

        tokens = [cls]
        pad_mask_cols = [False]    # CLS is never masked

        # Determine per-sample missing modalities
        # missing_modalities is a list[list[str]] of length B
        # Build a boolean mask (B, n_tokens) for the transformer key_padding_mask
        def _missing_for_sample(sample_idx: int, mod: str) -> bool:
            if missing_modalities is None:
                return False
            return mod in (missing_modalities[sample_idx] or [])

        # EEG token
        eeg_tok = (
            eeg_emb.unsqueeze(1) if eeg_emb is not None
            else torch.zeros(B, 1, self.embed_dim, device=device)
        )
        eeg_tok = eeg_tok + self.type_embedding(
            torch.tensor([MODALITY_IDX["eeg"]], device=device)
        )
        tokens.append(eeg_tok)

        # MRI token
        mri_tok = (
            mri_emb.unsqueeze(1) if mri_emb is not None
            else torch.zeros(B, 1, self.embed_dim, device=device)
        )
        mri_tok = mri_tok + self.type_embedding(
            torch.tensor([MODALITY_IDX["mri"]], device=device)
        )
        tokens.append(mri_tok)

        # HPO token
        hpo_tok = (
            hpo_emb.unsqueeze(1) if hpo_emb is not None
            else torch.zeros(B, 1, self.embed_dim, device=device)
        )
        hpo_tok = hpo_tok + self.type_embedding(
            torch.tensor([MODALITY_IDX["hpo"]], device=device)
        )
        tokens.append(hpo_tok)

        # Stack: (B, 4, 128) = [CLS, EEG, MRI, HPO]
        seq = torch.cat(tokens, dim=1)   # (B, 4, 128)

        # ----------------------------------------------------------------
        # Build key_padding_mask (B, 4) — True = ignore this position
        # ----------------------------------------------------------------
        key_pad = torch.zeros(B, seq.shape[1], dtype=torch.bool, device=device)
        # Col 0 = CLS (never masked)
        # Col 1 = EEG, Col 2 = MRI, Col 3 = HPO
        for b in range(B):
            for col_i, mod in enumerate(["eeg", "mri", "hpo"], start=1):
                is_missing = _missing_for_sample(b, mod)
                no_data = (
                    (mod == "eeg" and eeg_emb is None) or
                    (mod == "mri" and mri_emb is None) or
                    (mod == "hpo" and hpo_emb is None)
                )
                if is_missing or no_data:
                    key_pad[b, col_i] = True

        # ----------------------------------------------------------------
        # Fusion Transformer
        # ----------------------------------------------------------------
        fused = self.fusion_transformer(seq, src_key_padding_mask=key_pad)  # (B, 4, 128)
        cls_out = fused[:, 0, :]            # (B, 128) — CLS token

        # ----------------------------------------------------------------
        # Classification & severity heads
        # ----------------------------------------------------------------
        logits   = self.cls_head(cls_out)              # (B, 2)
        severity = self.severity_head(cls_out) * 100.0 # (B, 1) in [0, 100]

        # ----------------------------------------------------------------
        # Modality importance (interpretability)
        # ----------------------------------------------------------------
        importance = F.softmax(self.modality_importance, dim=0)  # (3,)

        return {
            "logits":              logits,
            "severity":            severity,
            "modality_importance": importance,
            "cls_embedding":       cls_out,
        }


def _infer_batch_size(batch: dict) -> int:
    """Get batch size from any tensor in the batch dict."""
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            return v.shape[0]
    return 1


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

def build_fusion_model(
    n_hpo: Optional[int] = None,
    freeze_encoders: bool = False,
    eeg_ckpt: Optional[str] = None,
    mri_ckpt: Optional[str] = None,
    hpo_ckpt: Optional[str] = None,
) -> LateFusionTransformer:
    """
    Build fusion model from config, optionally loading pretrained encoders.
    """
    c = cfg.model
    n_hpo = n_hpo or c.hpo_n_features
    return LateFusionTransformer(
        embed_dim=c.embed_dim,
        n_heads=c.fusion_heads,
        n_layers=c.fusion_layers,
        dropout=c.dropout,
        n_hpo=n_hpo,
        freeze_encoders=freeze_encoders,
        eeg_ckpt=eeg_ckpt,
        mri_ckpt=mri_ckpt,
        hpo_ckpt=hpo_ckpt,
    )
