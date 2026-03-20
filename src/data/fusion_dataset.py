"""
src/data/fusion_dataset.py
PyTorch Dataset that combines EEG, MRI, and HPO modalities.
Handles missing modalities gracefully for training and inference.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import cfg


class MultimodalDataset(Dataset):
    """
    Dataset that returns samples from any combination of EEG, MRI, HPO modalities.

    Each item returns a dict with available modality tensors + label + DQ.

    Parameters
    ----------
    eeg_records  : list of dicts with keys: subject_id, epoch_path|epochs, label, dq
    mri_records  : list of dicts with keys: participant_id, out_path|slices, label, dq
    hpo_records  : list of dicts with keys: disease, features (array), label, dq
    modality_dropout_p : float — probability of randomly dropping each modality
                         during training (set to 0 for eval/inference)
    augment      : bool — whether to apply augmentations
    """

    def __init__(
        self,
        eeg_records:  Optional[List[dict]] = None,
        mri_records:  Optional[List[dict]] = None,
        hpo_records:  Optional[List[dict]] = None,
        modality_dropout_p: float = 0.3,
        augment: bool = False,
        seed: int = 42,
    ):
        self.eeg_records = eeg_records or []
        self.mri_records = mri_records or []
        self.hpo_records = hpo_records or []
        self.modality_dropout_p = modality_dropout_p
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        # Build a unified list of samples
        # Each sample references whichever modalities it has data for
        self._samples = self._build_sample_list()

    # ------------------------------------------------------------------
    def _build_sample_list(self) -> List[dict]:
        """
        Merge all records into a unified list.
        If datasets have no overlapping subjects (as is the case here),
        each sample has only one or two modalities available.
        The fusion model is trained to handle missing modalities.
        """
        samples = []

        # EEG samples
        for rec in self.eeg_records:
            samples.append({
                "eeg_source": rec,
                "mri_source": None,
                "hpo_source": None,
                "label": int(rec.get("label", 0)),
                "dq": float(rec.get("dq", 90.0)),
                "subject_id": str(rec.get("subject_id", "")),
                "modalities": ["eeg"],
            })

        # MRI samples
        for rec in self.mri_records:
            samples.append({
                "eeg_source": None,
                "mri_source": rec,
                "hpo_source": None,
                "label": int(rec.get("label", 0)),
                "dq": float(rec.get("dq", 90.0)),
                "subject_id": str(rec.get("participant_id", "")),
                "modalities": ["mri"],
            })

        # HPO samples
        for rec in self.hpo_records:
            samples.append({
                "eeg_source": None,
                "mri_source": None,
                "hpo_source": rec,
                "label": int(rec.get("label", 0)),
                "dq": float(rec.get("dq", 75.0)),
                "subject_id": str(rec.get("disease", "")),
                "modalities": ["hpo"],
            })

        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------
    def _load_eeg(self, rec: dict) -> torch.Tensor:
        """Load EEG epoch tensor (19, 7680) from a single epoch or path."""
        if "epochs" in rec and rec["epochs"] is not None:
            epochs = rec["epochs"]
        elif "epoch_path" in rec and rec["epoch_path"] is not None:
            epochs = np.load(rec["epoch_path"], allow_pickle=False)
        else:
            return torch.zeros(cfg.model.eeg_channels, cfg.model.eeg_timesteps)

        epochs = epochs.astype(np.float32)  # (n_epochs, 19, T)
        # Sample a random epoch during training, first epoch for eval
        if self.augment and epochs.shape[0] > 1:
            idx = self.rng.integers(0, epochs.shape[0])
        else:
            idx = 0

        epoch = epochs[idx]  # (19, T)

        # Ensure correct shape
        T = cfg.model.eeg_timesteps
        C = cfg.model.eeg_channels

        if epoch.shape[0] < C:
            pad = np.zeros((C - epoch.shape[0], epoch.shape[1]), dtype=np.float32)
            epoch = np.vstack([epoch, pad])
        epoch = epoch[:C, :]

        if epoch.shape[1] < T:
            pad_w = T - epoch.shape[1]
            epoch = np.hstack([epoch, np.zeros((C, pad_w), dtype=np.float32)])
        epoch = epoch[:, :T]

        if self.augment:
            from src.data.eeg_loader import augment_eeg_epochs
            epoch = augment_eeg_epochs(epoch, rng=self.rng)

        return torch.from_numpy(epoch)   # (19, 7680)

    def _load_mri(self, rec: dict) -> torch.Tensor:
        """Load MRI slice tensor (3, 64, 64) from path or in-memory."""
        if "slices" in rec and rec["slices"] is not None:
            slices = rec["slices"].astype(np.float32)
        elif "out_path" in rec and rec["out_path"] is not None:
            slices = np.load(rec["out_path"], allow_pickle=False).astype(np.float32)
        else:
            return torch.zeros(cfg.model.mri_slices, cfg.model.mri_img_size, cfg.model.mri_img_size)

        if self.augment:
            from src.data.mri_loader import augment_mri_slices
            slices = augment_mri_slices(slices, rng=self.rng)

        return torch.from_numpy(slices)  # (3, 64, 64)

    def _load_hpo(self, rec: dict) -> torch.Tensor:
        """Load HPO feature vector from record."""
        if "features" in rec and rec["features"] is not None:
            feats = rec["features"].astype(np.float32)
        else:
            n = cfg.model.hpo_n_features
            return torch.zeros(n)
        return torch.from_numpy(feats)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]
        item: dict = {
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "dq":    torch.tensor(sample["dq"],    dtype=torch.float32),
            "missing_modalities": [],
        }

        # ---- EEG ----
        if sample["eeg_source"] is not None:
            drop = self.augment and self.rng.random() < self.modality_dropout_p
            if not drop:
                item["eeg"] = self._load_eeg(sample["eeg_source"])
            else:
                item["missing_modalities"].append("eeg")
        else:
            item["missing_modalities"].append("eeg")

        # ---- MRI ----
        if sample["mri_source"] is not None:
            drop = self.augment and self.rng.random() < self.modality_dropout_p
            if not drop:
                item["mri"] = self._load_mri(sample["mri_source"])
            else:
                item["missing_modalities"].append("mri")
        else:
            item["missing_modalities"].append("mri")

        # ---- HPO ----
        if sample["hpo_source"] is not None:
            drop = self.augment and self.rng.random() < self.modality_dropout_p
            if not drop:
                item["hpo"] = self._load_hpo(sample["hpo_source"])
            else:
                item["missing_modalities"].append("hpo")
        else:
            item["missing_modalities"].append("hpo")

        return item


def build_hpo_records(
    hpo_processed_dir: str | Path,
) -> List[dict]:
    """
    Load processed HPO arrays and return list of records for MultimodalDataset.
    """
    hpo_dir = Path(hpo_processed_dir)
    X    = np.load(str(hpo_dir / "hpo_matrix.npy"),       allow_pickle=False)
    y    = np.load(str(hpo_dir / "hpo_labels.npy"),        allow_pickle=False)
    dq   = np.load(str(hpo_dir / "hpo_dq.npy"),            allow_pickle=False)
    names = np.load(str(hpo_dir / "hpo_disease_names.npy"), allow_pickle=True)

    records = []
    for i in range(len(y)):
        records.append({
            "disease":  str(names[i]),
            "features": X[i],
            "label":    int(y[i]),
            "dq":       float(dq[i]),
        })
    return records


def build_eeg_records(
    eeg_processed_dir: str | Path,
    eeg_raw_dir: str | Path,
) -> List[dict]:
    """
    Build EEG records by scanning processed epoch files.
    Falls back to raw label parsing if needed.
    """
    proc_dir = Path(eeg_processed_dir)
    raw_dir  = Path(eeg_raw_dir)

    records = []
    for epoch_path in sorted(proc_dir.glob("*_epochs.npy")):
        sid = epoch_path.stem.replace("_epochs", "")
        feat_path = proc_dir / f"{sid}_features.npy"

        label, dq = 0, 90.0
        # Try to read label from a labels csv if saved
        label_csv = proc_dir / "labels.csv"
        if label_csv.exists():
            import pandas as pd
            ldf = pd.read_csv(str(label_csv))
            row = ldf[ldf["subject_id"].astype(str) == sid]
            if len(row) > 0:
                label = int(row.iloc[0].get("label", 0))
                dq    = float(row.iloc[0].get("dq", 90.0))

        records.append({
            "subject_id": sid,
            "epoch_path": str(epoch_path),
            "feat_path":  str(feat_path) if feat_path.exists() else None,
            "label": label,
            "dq":    dq,
        })
    return records


def build_mri_records(
    mri_processed_dir: str | Path,
) -> List[dict]:
    """
    Build MRI records from processed .npy slice files.
    """
    proc_dir = Path(mri_processed_dir)
    records = []
    for npy_path in sorted(proc_dir.glob("sub-*.npy")):
        pid = npy_path.stem
        records.append({
            "participant_id": pid,
            "out_path": str(npy_path),
            "label": 0,  # Baby Open Brains = healthy controls
            "dq":    90.0,
        })
    return records


def collate_multimodal(batch: List[dict]) -> dict:
    """
    Custom collate_fn for DataLoader.
    Handles variable missing modalities across samples in a batch.
    Pads missing modalities as zero tensors.
    """
    keys_present = set()
    for item in batch:
        if "eeg" in item: keys_present.add("eeg")
        if "mri" in item: keys_present.add("mri")
        if "hpo" in item: keys_present.add("hpo")

    result: dict = {
        "label": torch.stack([item["label"] for item in batch]),
        "dq":    torch.stack([item["dq"]    for item in batch]),
        "missing_modalities": [item["missing_modalities"] for item in batch],
    }

    if "eeg" in keys_present:
        C = cfg.model.eeg_channels
        T = cfg.model.eeg_timesteps
        eeg_tensors = [
            item.get("eeg", torch.zeros(C, T)) for item in batch
        ]
        result["eeg"] = torch.stack(eeg_tensors)

    if "mri" in keys_present:
        S = cfg.model.mri_slices
        H = W = cfg.model.mri_img_size
        mri_tensors = [
            item.get("mri", torch.zeros(S, H, W)) for item in batch
        ]
        result["mri"] = torch.stack(mri_tensors)

    if "hpo" in keys_present:
        n_hpo = cfg.model.hpo_n_features
        hpo_tensors = [
            item.get("hpo", torch.zeros(n_hpo)) for item in batch
        ]
        result["hpo"] = torch.stack(hpo_tensors)

    return result


def build_dataloaders(
    eeg_processed_dir: Optional[str | Path] = None,
    mri_processed_dir: Optional[str | Path] = None,
    hpo_processed_dir: Optional[str | Path] = None,
    eeg_raw_dir: Optional[str | Path] = None,
    batch_size: int = 16,
    modality_dropout_p: float = 0.3,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders from processed data directories.

    Uses HPO dataset for train/val/test split (largest dataset).
    EEG and MRI are added to all splits as single-modality samples.

    Returns: (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split

    hpo_records = []
    if hpo_processed_dir and Path(hpo_processed_dir).exists():
        hpo_records = build_hpo_records(hpo_processed_dir)

    eeg_records = []
    if eeg_processed_dir and Path(eeg_processed_dir).exists():
        eeg_records = build_eeg_records(
            eeg_processed_dir,
            eeg_raw_dir or cfg.paths.eeg_raw,
        )

    mri_records = []
    if mri_processed_dir and Path(mri_processed_dir).exists():
        mri_records = build_mri_records(mri_processed_dir)

    # Split HPO records (largest dataset) 70/15/15
    hpo_labels = [r["label"] for r in hpo_records] if hpo_records else []
    if len(hpo_records) > 0:
        hpo_train, hpo_temp = train_test_split(
            hpo_records, test_size=0.30, stratify=hpo_labels, random_state=seed
        )
        hpo_temp_labels = [r["label"] for r in hpo_temp]
        hpo_val, hpo_test = train_test_split(
            hpo_temp, test_size=0.50, stratify=hpo_temp_labels, random_state=seed
        )
    else:
        hpo_train = hpo_val = hpo_test = []

    def _make_loader(hpo_recs, augment: bool) -> DataLoader:
        ds = MultimodalDataset(
            eeg_records=eeg_records,
            mri_records=mri_records,
            hpo_records=hpo_recs,
            modality_dropout_p=modality_dropout_p if augment else 0.0,
            augment=augment,
            seed=seed,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=augment,
            collate_fn=collate_multimodal,
            num_workers=0,
            pin_memory=False,
        )

    train_loader = _make_loader(hpo_train, augment=True)
    val_loader   = _make_loader(hpo_val,   augment=False)
    test_loader  = _make_loader(hpo_test,  augment=False)

    return train_loader, val_loader, test_loader
