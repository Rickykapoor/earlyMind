"""
src/training/trainer.py
Full training loop with early stopping, gradient clipping, and scheduler.
Handles encoder pretraining phases and fusion training.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.config import cfg
from src.training.losses import CombinedLoss, FocalLoss


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor a validation metric and stop training when it stops improving."""

    def __init__(self, patience: int = 15, mode: str = "max", delta: float = 1e-4):
        self.patience = patience
        self.mode     = mode
        self.delta    = delta
        self.best     = float("-inf") if mode == "max" else float("inf")
        self.counter  = 0
        self.stopped  = False

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            (self.mode == "max" and metric > self.best + self.delta) or
            (self.mode == "min" and metric < self.best - self.delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                return True
        return False


# ---------------------------------------------------------------------------
# Encoder Pretraining — Generic
# ---------------------------------------------------------------------------

def pretrain_encoder(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    lr: float,
    device: torch.device,
    save_path: str,
    loss_fn: Callable,
    get_logits_fn: Callable,
    task_name: str = "encoder",
    patience: int = 10,
) -> Dict[str, List]:
    """
    Generic encoder pretraining loop.

    Parameters
    ----------
    encoder      : nn.Module with a forward() that returns dict
    train_loader : DataLoader yielding (inputs, labels) or dict
    val_loader   : DataLoader
    n_epochs     : int
    lr           : float
    device       : torch.device
    save_path    : str — where to save best checkpoint
    loss_fn      : callable(logits, labels) → scalar loss
    get_logits_fn: callable(encoder, batch) → logits tensor
    task_name    : str for logging

    Returns: history dict
    """
    encoder = encoder.to(device)
    optimizer = AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    es = EarlyStopping(patience=patience, mode="min")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        # ---- Train ----
        encoder.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = _unpack_batch(batch, device)
            logits = get_logits_fn(encoder, inputs)
            loss   = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ---- Validation ----
        encoder.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = _unpack_batch(batch, device)
                logits = get_logits_fn(encoder, inputs)
                loss   = loss_fn(logits, labels)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses) if val_losses else train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{task_name}] Epoch {epoch:3d}/{n_epochs} | "
                  f"train={train_loss:.4f} val={val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), save_path)

        if es.step(val_loss):
            print(f"  [{task_name}] Early stopping at epoch {epoch}")
            break

    print(f"  [{task_name}] Best val_loss = {best_val_loss:.4f} → {save_path}")
    return history


def _unpack_batch(batch, device):
    """Unpack (inputs, labels) or list/tuple for generic pretrain loops."""
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        inputs, labels = batch
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        labels = labels.to(device)
        return inputs, labels
    raise ValueError(f"Unexpected batch type: {type(batch)}")


# ---------------------------------------------------------------------------
# Fusion Training
# ---------------------------------------------------------------------------

def train_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    n_epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    severity_weight: float = 0.5,
    grad_clip: float = 1.0,
    freeze_encoders_epochs: int = 10,
) -> Dict:
    """
    Full fusion model training loop.

    Phase 1 (epochs 1–freeze_encoders_epochs): encoders frozen, only fusion layer trained.
    Phase 2 (after freeze_encoders_epochs): all weights fine-tuned together.

    Saves:
      checkpoints/fusion_model.pt        ← best model (by val AUC)
      checkpoints/fusion_epoch_{N}.pt    ← every 10 epochs
      reports/training_history.json
    """
    from sklearn.metrics import roc_auc_score

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = cfg.paths.reports
    reports_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    loss_fn = CombinedLoss(
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        severity_weight=severity_weight,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    es = EarlyStopping(patience=patience, mode="max")

    best_auc   = 0.0
    best_path  = str(save_dir / "fusion_model.pt")
    history    = {
        "train_loss": [], "val_loss": [],
        "train_auc":  [], "val_auc":  [],
        "focal_loss": [], "severity_loss": [],
    }

    for epoch in range(1, n_epochs + 1):
        # Phase transition: unfreeze encoders after freeze_encoders_epochs
        if epoch == freeze_encoders_epochs + 1:
            model.unfreeze_encoders()
            # Re-create optimizer with all params
            optimizer = AdamW(
                model.parameters(), lr=lr * 0.1, weight_decay=weight_decay
            )
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            print(f"  [fusion] Epoch {epoch}: Unfreezing all encoder weights")

        # ---- Train ----
        model.train()
        train_epoch_losses = []
        all_probs, all_labels = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            batch_dev = _move_batch_to_device(batch, device)
            missing   = batch_dev.get("missing_modalities", None)

            out    = model(batch_dev, missing_modalities=missing)
            labels = batch_dev["label"]
            dq     = batch_dev["dq"]

            losses = loss_fn(out["logits"], labels, out["severity"], dq)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_epoch_losses.append(losses["total"].item())
            probs = torch.softmax(out["logits"], dim=-1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        scheduler.step()

        train_loss = float(np.mean(train_epoch_losses))
        try:
            train_auc = float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else 0.5
        except Exception:
            train_auc = 0.5

        # ---- Validation ----
        model.eval()
        val_losses_list = []
        val_probs, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch_dev = _move_batch_to_device(batch, device)
                missing   = batch_dev.get("missing_modalities", None)
                out       = model(batch_dev, missing_modalities=missing)
                labels    = batch_dev["label"]
                dq        = batch_dev["dq"]

                losses = loss_fn(out["logits"], labels, out["severity"], dq)
                val_losses_list.append(losses["total"].item())

                probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
                val_probs.extend(probs.tolist())
                val_labels.extend(labels.cpu().numpy().tolist())

        val_loss = float(np.mean(val_losses_list)) if val_losses_list else train_loss
        try:
            val_auc = float(roc_auc_score(val_labels, val_probs)) if len(set(val_labels)) > 1 else 0.5
        except Exception:
            val_auc = 0.5

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [fusion] Epoch {epoch:3d}/{n_epochs} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"train_AUC={train_auc:.3f} val_AUC={val_auc:.3f}")
            # Save checkpoint
            ckpt_path = str(save_dir / f"fusion_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)

        if es.step(val_auc):
            print(f"  [fusion] Early stopping at epoch {epoch}. Best val_AUC={best_auc:.4f}")
            break

    # Save training history
    hist_path = str(reports_dir / "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"  [fusion] Training complete. Best AUC={best_auc:.4f} → {best_path}")
    return {"history": history, "best_auc": best_auc, "best_path": best_path}


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in batch dict to device, preserving non-tensor entries."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    return result
