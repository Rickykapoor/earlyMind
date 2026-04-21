"""
api/config.py
Environment-variable driven configuration for the EarlyMind API.
"""
from __future__ import annotations

import os
from pathlib import Path

N_HPO = 5284

CHECKPOINT_DIR = os.environ.get("EARLYMIND_CKPT_DIR", "checkpoints")
DEVICE = "cpu"

MAX_EEG_UPLOAD_MB = int(os.environ.get("EARLYMIND_MAX_EEG_MB", "50"))
MAX_MRI_UPLOAD_MB = int(os.environ.get("EARLYMIND_MAX_MRI_MB", "200"))

UVICORN_HOST = os.environ.get("UVICORN_HOST", "0.0.0.0")
UVICORN_PORT = int(os.environ.get("UVICORN_PORT", "8000"))
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8501"))

LOG_LEVEL = os.environ.get("EARLYMIND_LOG_LEVEL", "INFO").upper()

DQ_BANDS = [
    {"label": "Typical", "range": (85, 100)},
    {"label": "Borderline", "range": (70, 85)},
    {"label": "Mild ID Risk", "range": (55, 70)},
    {"label": "Moderate ID Risk", "range": (35, 55)},
    {"label": "Severe ID Risk", "range": (20, 35)},
    {"label": "Profound ID Risk", "range": (0, 20)},
]


def get_dq_label(dq: float) -> str:
    for band in DQ_BANDS:
        lo, hi = band["range"]
        if lo <= dq <= hi:
            return band["label"]
    return "Profound ID Risk"
