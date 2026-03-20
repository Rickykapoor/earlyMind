"""
src/training/losses.py
Focal Loss for classification and combined loss with DQ severity regression.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary and multi-class classification.
    Reduces the relative loss for well-classified examples to focus on hard cases.

    FL(p_t) = -alpha_t × (1 - p_t)^gamma × log(p_t)

    Parameters
    ----------
    alpha : float — weighting factor for rare class (ID positive)
    gamma : float — focusing parameter (0 = standard CE)
    reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C) for multi-class or (B, 1) for binary
        targets : (B,) long for multi-class or (B,) float for binary
        """
        if logits.shape[-1] == 1 or logits.ndim == 1:
            # Binary focal loss via BCE path
            logits = logits.squeeze(-1).float()
            targets = targets.float()
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            pt = torch.exp(-bce)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * (1 - pt) ** self.gamma * bce
        else:
            # Multi-class focal loss
            targets = targets.long()
            log_prob = F.log_softmax(logits, dim=-1)
            prob     = torch.exp(log_prob)

            # Gather target class probabilities
            pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
            log_pt = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)

            alpha_t = torch.where(targets == 1,
                                  torch.tensor(self.alpha, device=logits.device),
                                  torch.tensor(1 - self.alpha, device=logits.device))
            focal = -alpha_t * (1 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class SeverityLoss(nn.Module):
    """
    Smooth-L1 loss on DQ severity predictions.
    Masked to ID-positive samples only (label == 1).
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pred_dq: torch.Tensor,    # (B, 1)
        true_dq: torch.Tensor,    # (B,)
        labels:  torch.Tensor,    # (B,) — binary
    ) -> torch.Tensor:
        pred_dq = pred_dq.squeeze(-1).float()   # (B,)
        true_dq = true_dq.float()
        labels  = labels.long()

        # Mask to ID-positive samples
        mask = labels == 1
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_dq.device, requires_grad=True)

        loss = F.smooth_l1_loss(
            pred_dq[mask], true_dq[mask], beta=self.beta, reduction="mean"
        )
        return loss


class CombinedLoss(nn.Module):
    """
    Combined = FocalLoss(classification) + severity_weight × SeverityLoss(DQ).

    Parameters
    ----------
    focal_alpha    : float
    focal_gamma    : float
    severity_weight: float — weight for DQ loss term
    pos_weight     : float — optional BCE pos_weight (overrides alpha for binary)
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        severity_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.severity = SeverityLoss()
        self.severity_weight = severity_weight
        self.pos_weight = pos_weight   # for HPO class imbalance

    def forward(
        self,
        logits:   torch.Tensor,    # (B, 2) classification logits
        labels:   torch.Tensor,    # (B,) long
        pred_dq:  torch.Tensor,    # (B, 1)
        true_dq:  torch.Tensor,    # (B,)
    ) -> dict:
        focal_loss    = self.focal(logits, labels)
        severity_loss = self.severity(pred_dq, true_dq, labels)
        total         = focal_loss + self.severity_weight * severity_loss

        return {
            "total":         total,
            "focal":         focal_loss.detach(),
            "severity":      severity_loss.detach(),
        }
