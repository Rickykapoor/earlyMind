# EarlyMind 🧠
### Multimodal Infant ID Risk Detection — Deployed

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

---

## About this Space

EarlyMind is a production-ready multimodal deep learning system for **early detection of Intellectual Disability (ID) risk** in infants aged 0–36 months. It fuses three data modalities:

| Modality | Data | Encoder |
|----------|------|---------|
| **EEG** | 19-channel neonatal EEG | Multi-scale CNN + Transformer |
| **MRI** | T1w/T2w brain MRI (3 slices) | EfficientNet-B0 + cross-slice attention |
| **HPO** | Human Phenotype Ontology (5,284 terms) | Self-attention MLP |
| | | LateFusionTransformer → ID Risk + DQ |

**API base URL**: `http://localhost:8000` (FastAPI)  
**UI URL**: `http://localhost:8501` (Streamlit)

---

## Quick Start

### Local Development (Docker)

```bash
docker compose up --build
```

Then open:
- **Streamlit UI**: http://localhost:8501
- **API docs**: http://localhost:8000/docs

### Local Development (Python)

```bash
conda env create -f environment.yml
conda activate infant_id
bash startup.sh
```

### HuggingFace Spaces

This Space is deployed via Docker. To deploy your own:

1. Fork this repository
2. Create a new Space at https://huggingface.co/new-space
3. Select **Docker** as the SDK
4. Set the Docker image to your GitHub Container Registry URL (e.g. `ghcr.io/username/earlymind:latest`)
5. The Space will auto-deploy from your Dockerfile

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + model status |
| `GET` | `/model/info` | Model metadata + DQ severity bands |
| `POST` | `/predict` | Full multimodal prediction |
| `POST` | `/predict/eeg` | EEG-only prediction |
| `POST` | `/predict/mri` | MRI-only prediction |
| `POST` | `/predict/hpo` | HPO-only prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 32) |
| `POST` | `/preprocess/edf` | Upload EDF → extract features |
| `POST` | `/preprocess/nifti` | Upload NIfTI → extract slices |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hpo": [0.0, 0.1, 0.5, 0.0],
    "hpo_symptom_score": 0.3
  }'
```

### Example Response

```json
{
  "risk_probability": 0.42,
  "dq_estimate": 65.0,
  "dq_label": "Mild ID Risk",
  "confidence": "Moderate",
  "modality_importance": [0.1, 0.1, 0.8],
  "warnings": ["HPO encoder undertrained; result blended with clinical score."]
}
```

### Upload EDF for preprocessing

```bash
curl -X POST http://localhost:8000/preprocess/edf \
  -F "file=@path/to/recording.edf"
```

---

## DQ Severity Scale

| DQ Range | Label |
|----------|-------|
| 85–100 | Typical development |
| 70–84 | Borderline — monitor |
| 55–69 | Mild ID risk |
| 35–54 | Moderate ID risk |
| 20–34 | Severe ID risk |
| 0–19 | Profound ID risk |

---

## ⚕️ Clinical Disclaimer

EarlyMind is a **research screening tool only**. It is not FDA cleared and does not provide clinical diagnosis. All results must be interpreted by qualified healthcare professionals.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EARLYMIND_CKPT_DIR` | `checkpoints` | Path to model checkpoints |
| `EARLYMIND_DEVICE` | `cpu` | Device for inference |
| `EARLYMIND_LOG_LEVEL` | `INFO` | Logging level |
| `UVICORN_PORT` | `8000` | FastAPI port |
| `STREAMLIT_PORT` | `8501` | Streamlit port |
| `EARLYMIND_MAX_EEG_MB` | `50` | Max EDF upload size (MB) |
| `EARLYMIND_MAX_MRI_MB` | `200` | Max NIfTI upload size (MB) |
