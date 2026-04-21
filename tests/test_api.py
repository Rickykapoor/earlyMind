"""
tests/test_api.py
Smoke tests for the EarlyMind API.
Run: pytest tests/test_api.py -v
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.config import get_dq_label
from api.schemas import PredictionRequest


class TestSchemas:
    def test_prediction_request_hpo_only(self):
        req = PredictionRequest(hpo=[0.1] * 5284, hpo_symptom_score=0.5)
        assert req.hpo is not None
        assert len(req.hpo) == 5284
        assert req.hpo_symptom_score == 0.5

    def test_prediction_request_all_modalities(self):
        req = PredictionRequest(
            eeg=[[0.0] * 7680] * 19,
            mri=[[[0.0] * 64] * 64] * 3,
            hpo=[0.0] * 5284,
        )
        assert req.eeg is not None
        assert req.mri is not None
        assert req.hpo is not None

    def test_prediction_request_null_modalities(self):
        req = PredictionRequest()
        assert req.eeg is None
        assert req.mri is None
        assert req.hpo is None

    def test_hpo_truncated_if_too_long(self):
        with pytest.raises(ValueError):
            PredictionRequest(hpo=[0.1] * 6000)

    def test_eeg_wrong_ndim_rejected(self):
        with pytest.raises(ValueError):
            PredictionRequest(eeg=[[[0.0] * 100] * 5] * 3)

    def test_mri_wrong_slices_rejected(self):
        with pytest.raises(ValueError):
            PredictionRequest(mri=[[[0.0] * 64] * 64] * 5)


class TestConfig:
    def test_dq_label_typical(self):
        assert get_dq_label(90.0) == "Typical"

    def test_dq_label_borderline(self):
        assert get_dq_label(75.0) == "Borderline"

    def test_dq_label_mild(self):
        assert get_dq_label(60.0) == "Mild ID Risk"

    def test_dq_label_moderate(self):
        assert get_dq_label(45.0) == "Moderate ID Risk"

    def test_dq_label_severe(self):
        assert get_dq_label(25.0) == "Severe ID Risk"

    def test_dq_label_profound(self):
        assert get_dq_label(10.0) == "Profound ID Risk"


class TestAPIImports:
    def test_api_config_imports(self):
        from api.config import CHECKPOINT_DIR, DEVICE, N_HPO, DQ_BANDS
        assert N_HPO == 5284
        assert DEVICE == "cpu"
        assert len(DQ_BANDS) == 6

    def test_api_schemas_imports(self):
        from api.schemas import (
            PredictionRequest, PredictionResponse,
            HealthResponse, ModelInfoResponse,
            EDFPreprocessResponse, NIfTIPreprocessResponse,
        )
        assert PredictionRequest is not None
        assert PredictionResponse is not None

    def test_api_main_imports(self):
        from api.main import app
        assert app is not None
        assert app.title == "EarlyMind API"
