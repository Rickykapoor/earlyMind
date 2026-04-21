"""
api/main.py
FastAPI application for the EarlyMind multimodal ID risk prediction API.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import mne
import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import (
    CHECKPOINT_DIR,
    DEVICE,
    DQ_BANDS,
    MAX_EEG_UPLOAD_MB,
    MAX_MRI_UPLOAD_MB,
    N_HPO,
    UVICORN_HOST,
    UVICORN_PORT,
    get_dq_label,
)
from api.inference import load_model, run_prediction
from api.preprocessing import preprocess_edf, preprocess_nifti
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DQBand,
    EDFPreprocessResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    NIfTIPreprocessResponse,
    PredictionRequest,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EarlyMind API",
    description="Multimodal deep learning API for early detection of infant Intellectual Disability risk. "
                "Fuses EEG, MRI, and HPO phenotype data.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("EarlyMind API starting up...")
    model = load_model()
    if model is not None:
        logger.info("Fusion model loaded successfully.")
    else:
        logger.warning("Model not loaded — /predict endpoints will return 503.")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    model = load_model()
    if model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device=DEVICE,
            model_info={},
        )
    ckpt = Path(CHECKPOINT_DIR) / "fusion_model.pt"
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=DEVICE,
        model_info={
            "checkpoint": str(ckpt),
            "size_mb": round(ckpt.stat().st_size / (1024 * 1024), 1) if ckpt.exists() else 0,
        },
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    return ModelInfoResponse(
        model_name="EarlyMind LateFusionTransformer",
        checkpoint_path=str(Path(CHECKPOINT_DIR) / "fusion_model.pt"),
        embed_dim=128,
        fusion_heads=4,
        fusion_layers=3,
        dropout=0.2,
        hpo_n_features=N_HPO,
        severity_bands=[DQBand(label=b["label"], range=b["range"]) for b in DQ_BANDS],
        supported_modalities=["eeg", "mri", "hpo"],
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, responses={503: {"model": ErrorResponse}}, tags=["Inference"])
async def predict(request: PredictionRequest):
    model = load_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Ensure checkpoints/fusion_model.pt exists.",
        )

    prob, dq, importance, confidence, warnings = run_prediction(
        eeg=request.eeg,
        mri=request.mri,
        hpo=request.hpo,
        hpo_symptom_score=request.hpo_symptom_score,
        eeg_symptom_score=request.eeg_symptom_score,
        mri_symptom_score=request.mri_symptom_score,
    )

    if prob is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(warnings),
        )

    return PredictionResponse(
        risk_probability=round(prob, 4),
        dq_estimate=round(dq, 1),
        dq_label=get_dq_label(dq),
        confidence=confidence,
        modality_importance=[round(float(v), 4) for v in importance],
        warnings=warnings,
    )


@app.post("/predict/eeg", response_model=PredictionResponse, responses={503: {"model": ErrorResponse}}, tags=["Inference"])
async def predict_eeg(eeg: list[list[float]], eeg_symptom_score: float = 0.0):
    return await predict(PredictionRequest(eeg=eeg, eeg_symptom_score=eeg_symptom_score))


@app.post("/predict/mri", response_model=PredictionResponse, responses={503: {"model": ErrorResponse}}, tags=["Inference"])
async def predict_mri(mri: list[list[list[float]]], mri_symptom_score: float = 0.0):
    return await predict(PredictionRequest(mri=mri, mri_symptom_score=mri_symptom_score))


@app.post("/predict/hpo", response_model=PredictionResponse, responses={503: {"model": ErrorResponse}}, tags=["Inference"])
async def predict_hpo(hpo: list[float], hpo_symptom_score: float = 0.0):
    return await predict(PredictionRequest(hpo=hpo, hpo_symptom_score=hpo_symptom_score))


@app.post("/predict/batch", response_model=BatchPredictionResponse, responses={503: {"model": ErrorResponse}}, tags=["Inference"])
async def predict_batch(request: BatchPredictionRequest):
    results = []
    for subject in request.subjects:
        result = await predict(subject)
        results.append(result)
    return BatchPredictionResponse(predictions=results)


@app.post("/preprocess/edf", response_model=EDFPreprocessResponse, responses={400: {"model": ErrorResponse}}, tags=["Preprocessing"])
async def preprocess_edf_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ("application/octet-stream", "application/edf", "application/x-edf") and not file.filename.endswith(".edf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be EDF format (.edf)")

    size_mb = file.size / (1024 * 1024) if file.size else 0
    if size_mb > MAX_EEG_UPLOAD_MB:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"File too large: {size_mb:.1f}MB > {MAX_EEG_UPLOAD_MB}MB limit")

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        n_epochs, n_channels, sfreq, duration, features, summary = preprocess_edf(tmp_path)
        return EDFPreprocessResponse(
            n_epochs=n_epochs,
            n_channels=n_channels,
            sample_rate=sfreq,
            duration_sec=round(duration, 2),
            features=features.tolist(),
            epochs_summary=summary,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"EDF processing failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/preprocess/nifti", response_model=NIfTIPreprocessResponse, responses={400: {"model": ErrorResponse}}, tags=["Preprocessing"])
async def preprocess_nifti_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ("application/octet-stream", "application/nifti-image") and not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be NIfTI format (.nii or .nii.gz)")

    size_mb = file.size / (1024 * 1024) if file.size else 0
    if size_mb > MAX_MRI_UPLOAD_MB:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"File too large: {size_mb:.1f}MB > {MAX_MRI_UPLOAD_MB}MB limit")

    suffix = ".nii.gz" if file.filename.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        subject_id, shape, n_slices, slices_shape, myelination_note = preprocess_nifti(tmp_path)
        return NIfTIPreprocessResponse(
            subject_id=subject_id,
            shape=shape,
            n_slices=n_slices,
            slices_shape=slices_shape,
            myelination_note=myelination_note,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"NIfTI processing failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
