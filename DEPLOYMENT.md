# EarlyMind Deployment Guide

## Local Development

### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
docker compose up --build

# Services:
#   Streamlit UI:  http://localhost:8501
#   FastAPI:       http://localhost:8000
#   API docs:      http://localhost:8000/docs
```

### Option 2: Manual (without Docker)

```bash
# Terminal 1: FastAPI backend
uvicorn api.main:app --port 8000 --reload

# Terminal 2: Streamlit frontend
streamlit run app.py --server.port 8501

# Or run both via startup script
bash startup.sh
```

## HuggingFace Spaces Deployment

### Prerequisites

1. Push code to GitHub
2. Create HF Space: https://huggingface.co/new-space
   - SDK: **Docker**
   - Hardware: **CPU** (or **GPU T4** if you need GPU)
3. Connect HF Space to your GitHub repo (Settings → Repository → Link to GitHub)
4. Add GitHub secrets:
   - `HF_TOKEN`: HuggingFace write token (with write access)

### Deploy Steps

```bash
# Push to main — HF auto-builds from Dockerfile
git push origin main
```

The `startup.sh` script runs automatically inside the HF container.

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Full Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hpo": [0.0] * 5284,
    "hpo_symptom_score": 0.3,
    "eeg_symptom_score": 0.1,
    "mri_symptom_score": 0.2
  }'
```

### Upload EDF for Preprocessing

```bash
curl -X POST http://localhost:8000/preprocess/edf \
  -F "file=@datasets/eeg/helsinki_neonatal/1.edf"
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "subjects": [
      {"hpo": [0.0] * 5284, "hpo_symptom_score": 0.1},
      {"hpo": [0.5] * 5284, "hpo_symptom_score": 0.8}
    ]
  }'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EARLYMIND_API_URL` | `http://localhost:8000` | FastAPI base URL (Streamlit uses this) |
| `EARLYMIND_CKPT_DIR` | `checkpoints` | Model checkpoint directory |
| `EARLYMIND_DEVICE` | `cpu` | CPU or cuda |
| `EARLYMIND_LOG_LEVEL` | `INFO` | Logging level |
| `EARLYMIND_MAX_EEG_MB` | `50` | Max EDF file size (MB) |
| `EARLYMIND_MAX_MRI_MB` | `200` | Max NIfTI file size (MB) |

## Adding Checkpoints

### Option 1: Volume Mount (local Docker)

```yaml
# docker-compose.yml — add this to volumes
volumes:
  - /path/to/your/checkpoints:/app/checkpoints:ro
```

### Option 2: DVC Pull

```bash
dvc pull
```

### Option 3: Copy into Docker image (for HF Spaces)

Since HF Spaces doesn't persist large files, your checkpoints must be fetched at runtime:

```dockerfile
# Add to Dockerfile before the startup entrypoint:
RUN dvc pull || echo "Note: Run dvc pull after container starts to fetch checkpoints"
```

## CI/CD Pipeline

GitHub Actions automatically:
1. Runs pytest smoke tests
2. Builds Docker image
3. Pushes to GHCR (`ghcr.io/<owner>/<repo>`)

To trigger HF Space update after Docker push:
```bash
# Using hf_hub library
python -c "from huggingface_hub import HfApi; HfApi().create_space(repo_id='<owner>/earlymind', repo_type='space', exists_ok=True)"
```
