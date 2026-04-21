"""
api/schemas.py
Pydantic request/response schemas for the EarlyMind FastAPI.
"""
from __future__ import annotations

from typing import Annotated, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

N_HPO = 5284


class PredictionRequest(BaseModel):
    eeg: Annotated[
        Optional[List[List[float]]],
        Field(description="EEG epoch tensor (channels, timesteps). Provide null if unavailable.", examples=[None])
    ] = None
    mri: Annotated[
        Optional[List[List[List[float]]]],
        Field(description="MRI slices tensor (3, 64, 64). Provide null if unavailable.", examples=[None])
    ] = None
    hpo: Annotated[
        Optional[List[float]],
        Field(description=f"HPO feature vector ({N_HPO},). Provide null if unavailable.", examples=[None])
    ] = None
    hpo_symptom_score: Annotated[float, Field(default=0.0, ge=0.0, le=1.0, description="Clinical HPO symptom severity 0–1")] = 0.0
    eeg_symptom_score: Annotated[float, Field(default=0.0, ge=0.0, le=1.0, description="Clinical EEG symptom severity 0–1")] = 0.0
    mri_symptom_score: Annotated[float, Field(default=0.0, ge=0.0, le=1.0, description="Clinical MRI symptom severity 0–1")] = 0.0

    model_config = ConfigDict(extra="forbid")

    @field_validator("eeg")
    @classmethod
    def validate_eeg_shape(cls, v):
        if v is None:
            return None
        arr = np.array(v, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"EEG must be 2D (channels, timesteps), got {arr.ndim}D")
        return v

    @field_validator("mri")
    @classmethod
    def validate_mri_shape(cls, v):
        if v is None:
            return None
        arr = np.array(v, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"MRI must be 3D (slices, H, W), got {arr.ndim}D")
        if arr.shape[0] != 3:
            raise ValueError(f"MRI must have 3 slices, got {arr.shape[0]}")
        return v

    @field_validator("hpo")
    @classmethod
    def validate_hpo_length(cls, v):
        if v is None:
            return None
        if len(v) > N_HPO:
            raise ValueError(f"HPO vector too long: {len(v)} > {N_HPO}")
        return v


class PredictionResponse(BaseModel):
    risk_probability: Annotated[float, Field(ge=0.0, le=1.0, description="ID risk probability 0–1")]
    dq_estimate: Annotated[float, Field(ge=0.0, le=100.0, description="Developmental Quotient estimate 0–100")]
    dq_label: Annotated[str, Field(description="Human-readable DQ severity label")]
    confidence: Annotated[str, Field(description="'High', 'Moderate', or 'Low' based on distance from 0.5")]
    modality_importance: Annotated[List[float], Field(description="softmax weights over [EEG, MRI, HPO]")]
    warnings: Annotated[List[str], Field(default_factory=list, description="Model or data warnings")]

    model_config = ConfigDict(extra="forbid")


class BatchPredictionRequest(BaseModel):
    subjects: Annotated[List[PredictionRequest], Field(min_length=1, max_length=32)]
    model_config = ConfigDict(extra="forbid")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    model_config = ConfigDict(extra="forbid")


class EDFPreprocessResponse(BaseModel):
    n_epochs: Annotated[int, Field(ge=0, description="Number of 30s epochs extracted")]
    n_channels: Annotated[int, Field(description="Number of EEG channels")]
    sample_rate: Annotated[int, Field(description="Sampling frequency (Hz)")]
    duration_sec: Annotated[float, Field(description="Total recording duration (s)")]
    features: Annotated[List[float], Field(description="11 frequency-domain features: Delta, Theta, Alpha, Beta, TotalPower, BSR, IBI_Mean, IBI_Std, SEF95, Amp_Mean, Amp_Std")]
    epochs_summary: Annotated[str, Field(description="Human-readable summary of extracted epochs")]

    model_config = ConfigDict(extra="forbid")


class NIfTIPreprocessResponse(BaseModel):
    subject_id: Annotated[str, Field(description="Subject identifier from filename")]
    shape: Annotated[List[int], Field(description="Original volume shape (D, H, W)")]
    n_slices: Annotated[int, Field(description="Number of slices extracted")]
    slices_shape: Annotated[List[int], Field(description="Shape of each extracted slice (H, W)")]
    myelination_note: Annotated[str, Field(description="Note about myelination status")]

    model_config = ConfigDict(extra="forbid")


class DQBand(BaseModel):
    label: str
    range: tuple[int, int]


class ModelInfoResponse(BaseModel):
    model_name: str
    checkpoint_path: str
    embed_dim: int
    fusion_heads: int
    fusion_layers: int
    dropout: float
    hpo_n_features: int
    severity_bands: List[DQBand]
    supported_modalities: List[str]
    version: str

    model_config = ConfigDict(extra="forbid")


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    device: str
    model_info: dict

    model_config = ConfigDict(extra="forbid")


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

    model_config = ConfigDict(extra="forbid")
