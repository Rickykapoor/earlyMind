"""
api/inference.py
Model loading and prediction logic for the EarlyMind FastAPI.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from api.config import CHECKPOINT_DIR, DEVICE, N_HPO, get_dq_label

logger = logging.getLogger(__name__)

_model_instance = None


def load_model() -> Optional[torch.nn.Module]:
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    ckpt_path = Path(CHECKPOINT_DIR) / "fusion_model.pt"
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint not found at {ckpt_path}")
        return None

    import sys
    src_path = Path(__file__).resolve().parents[1]
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        from src.models.fusion_model import build_fusion_model
        model = build_fusion_model(n_hpo=N_HPO)
        state = torch.load(str(ckpt_path), map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        model.to(DEVICE)
        model.eval()
        _model_instance = model
        logger.info(f"Model loaded from {ckpt_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def _pad_tensor(arr: np.ndarray, target_len: int, axis: int = -1) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32)
    current_len = arr.shape[axis]
    if current_len < target_len:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, target_len - current_len)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
    elif current_len > target_len:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, target_len)
        arr = arr[tuple(slices)]
    return arr


def _build_batch(
    eeg: Optional[List[List[float]]],
    mri: Optional[List[List[List[float]]]],
    hpo: Optional[List[float]],
) -> tuple[Dict[str, torch.Tensor], List[str]]:
    batch: Dict[str, torch.Tensor] = {}
    missing: List[str] = []

    if eeg is not None:
        arr = _pad_tensor(np.array(eeg), 7680, axis=1)
        arr = _pad_tensor(arr, 19, axis=0)
        batch["eeg"] = torch.from_numpy(arr).unsqueeze(0)
    else:
        missing.append("eeg")

    if mri is not None:
        arr = np.array(mri, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.unsqueeze(1) if hasattr(arr, "unsqueeze") else np.expand_dims(arr, axis=1)
        arr = np.squeeze(arr, axis=1) if arr.ndim == 4 and arr.shape[1] == 1 else arr
        batch["mri"] = torch.from_numpy(arr).unsqueeze(0).unsqueeze(2)
    else:
        missing.append("mri")

    if hpo is not None:
        arr = _pad_tensor(np.array(hpo), N_HPO, axis=0)
        batch["hpo"] = torch.from_numpy(arr).unsqueeze(0)
    else:
        missing.append("hpo")

    return batch, missing


def run_prediction(
    eeg: Optional[List[List[float]]] = None,
    mri: Optional[List[List[List[float]]]] = None,
    hpo: Optional[List[float]] = None,
    hpo_symptom_score: float = 0.0,
    eeg_symptom_score: float = 0.0,
    mri_symptom_score: float = 0.0,
) -> tuple[Optional[float], Optional[float], Optional[List[float]], Optional[str], List[str]]:
    model = load_model()
    if model is None:
        return None, None, None, None, ["Model not loaded. Train and place checkpoints/fusion_model.pt."]

    batch, missing = _build_batch(eeg, mri, hpo)
    if len(batch) == 0:
        return None, None, None, None, ["At least one modality (EEG, MRI, or HPO) must be provided."]

    warnings: List[str] = []
    if "eeg" not in batch:
        warnings.append("EEG encoder undertrained; result blended with clinical score.")
    if "mri" not in batch:
        warnings.append("MRI encoder undertrained; result blended with clinical score.")
    if "hpo" not in batch:
        warnings.append("HPO encoder undertrained; result blended with clinical score.")

    with torch.no_grad():
        out = model(batch, missing_modalities=[missing])
        prob = float(F.softmax(out["logits"], dim=-1)[0, 1].cpu())
        dq = float(out["severity"][0, 0].cpu().item())
        importance = out["modality_importance"].cpu().numpy().tolist()

    combined_score = max(hpo_symptom_score, eeg_symptom_score, mri_symptom_score)
    if combined_score > 0:
        prob = 0.3 * prob + 0.7 * combined_score
        dq = max(0.0, dq - combined_score * 60.0)

    dist = abs(prob - 0.5)
    if dist > 0.3:
        confidence = "High"
    elif dist > 0.15:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return prob, dq, importance, confidence, warnings
