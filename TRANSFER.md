# EarlyMind — Transfer Guide (Mac → Windows)

> This guide walks you through transferring the EarlyMind project to a Windows machine and verifying that Google Colab notebooks run end-to-end.

---

## Part A — On the Mac (Sending Side)

### A1. Upload Large Files to GitHub Releases

Large model checkpoints and raw datasets **cannot** be committed to GitHub (>100 MB limit). They are hosted as **GitHub Release assets** under tag `v1.0.0-data`.

**Files to upload:**

| File | Size | Purpose |
|------|------|---------|
| `checkpoints/fusion_model.pt` | ~485 MB | Main inference model |
| `checkpoints/hpo_encoder_pretrained.pt` | ~437 MB | HPO feature encoder |
| `checkpoints/mri_encoder_pretrained.pt` | ~42 MB | MRI feature encoder |
| `checkpoints/eeg_encoder_pretrained.pt` | ~3.3 MB | EEG feature encoder |
| `eeg_raw.tar.gz` | ~83 MB | Helsinki Neonatal EEG dataset |
| `mri_raw.tar.gz` | ~624 MB | Baby Open Brains MRI dataset |
| `facial_raw.tar.gz` | ~5.8 MB | HPO facial phenotype data |

**Steps:**
1. Go to your GitHub repo → **Releases** → **Draft a new release**
2. Tag: `v1.0.0-data`
3. Title: `Data Release v1.0`
4. Drag and drop all 7 files above into the release assets area
5. Click **Publish release**

### A2. Push Source Code

```bash
cd /Users/vanshtuli/Desktop/Archive

git init                         # skip if already a git repo
git remote add origin https://github.com/YOUR_USERNAME/earlyMind.git

# Stage everything (large files excluded by .gitignore)
git add .
git status                       # verify no .pt or .tar.gz files are staged

git commit -m "chore: add Windows scripts and transfer files"
git push -u origin main
```

### A3. Verify What Was Pushed

```bash
git ls-files | grep -E '\.(pt|tar\.gz|zip)$'
# Should return NOTHING — these must NOT be in the repo
```

---

## Part B — On the Windows Machine (Receiving Side)

### B1. Install Prerequisites

| Tool | Download |
|------|---------|
| **Anaconda** (Python + conda) | https://www.anaconda.com/download |
| **Git** | https://git-scm.com/download/win |
| Windows 10/11 (for built-in `curl` and `tar`) | — |

### B2. Clone the Repository

Open **Anaconda Prompt** (or any cmd/PowerShell with conda in PATH):

```batch
git clone https://github.com/YOUR_USERNAME/earlyMind.git
cd earlyMind
```

### B3. Run Setup

```batch
setup.bat
```

This will automatically:
- ✅ Create the `infant_id` conda environment from `environment.yml`
- ✅ Install all pip packages from `requirements.txt`
- ✅ Verify all Python imports
- ✅ Create all required output directories
- ✅ Download all 7 large files from GitHub Releases (~1.2 GB total)
- ✅ Extract the dataset archives into the correct folders

> **Estimated time:** 15–40 minutes depending on internet speed (downloads ~1.2 GB)

### B4. Launch the Application

```batch
start_windows.bat
```

This opens two separate console windows:
- **FastAPI** on http://localhost:8000/docs (Swagger UI)
- **Streamlit** on http://localhost:8501 (main UI)

The browser will open `http://localhost:8501` automatically.

### B5. Verify Installation

Open **Anaconda Prompt** and run:

```batch
call conda activate infant_id

REM 1. Check imports
python -c "import torch, mne, nibabel, streamlit; print('All OK')"

REM 2. Check config loads
python -c "from src.config import cfg; print(cfg)"

REM 3. Health check (while app is running)
curl http://localhost:8000/health

REM 4. Run test suite
python -m pytest tests\ -v
```

---

## Part C — Google Colab Notebooks

The 6 notebooks in `notebooks/` are **fully self-contained** for Colab. Each one:

| Cell | What it does |
|------|-------------|
| Cell 1 | Clones this repo from GitHub |
| Cell 2 | Installs all pip dependencies |
| Cell 3 | Checks for GPU (T4/A100 on Colab) |
| Cell 4 | Downloads raw datasets from GitHub Releases |
| Cell 5+ | Runs preprocessing / training |
| Last cell | Commits results back to GitHub |

### Run Order (must be sequential!)

```
notebooks/01_eeg_preprocess.ipynb   ← start here
notebooks/02_mri_preprocess.ipynb
notebooks/03_hpo_preprocess.ipynb
notebooks/04_train_encoders.ipynb
notebooks/05_fusion_train.ipynb
notebooks/06_evaluate.ipynb         ← finish here
```

### How to Open in Colab

1. Go to https://colab.research.google.com
2. **File → Open notebook → GitHub**
3. Enter: `https://github.com/YOUR_USERNAME/earlyMind`
4. Select `notebooks/01_eeg_preprocess.ipynb`
5. **Runtime → Change runtime type → T4 GPU**
6. **Runtime → Run all**

> **Tip:** Use **Colab Pro** for the MRI training notebook (`05_fusion_train.ipynb`) — it requires > 4 GB GPU VRAM and can take 30–60 min on a free T4.

---

## Environment Variables (Optional)

Set these in Windows System Properties → Environment Variables if you want to customize behaviour:

| Variable | Default | Description |
|----------|---------|-------------|
| `EARLYMIND_CKPT_DIR` | `checkpoints` | Path to model checkpoints |
| `EARLYMIND_DEVICE` | `cpu` | `cpu` or `cuda` |
| `EARLYMIND_LOG_LEVEL` | `INFO` | Logging verbosity |
| `EARLYMIND_MAX_EEG_MB` | `50` | Max EDF upload size (MB) |
| `EARLYMIND_MAX_MRI_MB` | `200` | Max NIfTI upload size (MB) |

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `conda: command not found` | Reinstall Anaconda, check "Add to PATH" during install |
| `curl` download fails | Check internet / firewall; manually download and place in the right folder |
| `tar` not found | Update to Windows 10 1803+ or install 7-Zip and extract manually |
| Port 8501 already in use | Kill the process: `netstat -ano \| findstr :8501`, then `taskkill /PID <PID> /F` |
| Colab Cell 1 fails (git clone) | Ensure the GitHub repo is **public**, or add a Personal Access Token |
| `ModuleNotFoundError: torch` | Run `pip install torch --index-url https://download.pytorch.org/whl/cu118` in the env |
| Checkpoints not found on startup | Verify `checkpoints\fusion_model.pt` exists; re-run `setup.bat` if missing |

---

## Post-Transfer Retraining Roadmap

> [!IMPORTANT]
> The checkpoints bundled in GitHub Releases (`v1.0.0-data`) were trained on the
> **pre-augmentation** dataset (Mar 21). The augmented MRI data (10,000 samples, Apr 12)
> is **not** in those checkpoints yet. Use this roadmap to retrain after transfer.

### What's Already There After Transfer

| Asset | Status |
|-------|--------|
| Raw datasets (EEG, MRI, HPO) | ✅ Downloaded by `setup.bat` from GitHub Releases |
| Stale checkpoints (Mar 21) | ✅ Downloaded — work for inference immediately |
| Augmented MRI data | ❌ Must be **regenerated** (deterministic, seed=42) |

### Retraining Steps — On Windows (local GPU)

Open **Anaconda Prompt**, `cd` to the project folder, then:

```batch
call conda activate infant_id

REM Step 1: Regenerate augmented MRI dataset (deterministic, seed=42 — same result as Mac)
python scripts/balance_mri.py --skip-preprocess

REM Step 2: Retrain all 3 encoders on augmented data
python src/scripts/run_eeg_encoder.py
python src/scripts/run_mri_encoder.py
python src/scripts/run_hpo_encoder.py

REM Step 3: Retrain fusion model with new encoder checkpoints
python src/scripts/run_fusion.py
```

> Estimated time on a mid-range GPU (RTX 3060+): ~2-4 hours for full pipeline.

### Retraining Steps — On Google Colab (alternative)

Run notebooks in order — they auto-regenerate augmented data and retrain:

```
notebooks/02_mri_preprocess.ipynb   ← re-runs augmentation (Cell 7)
notebooks/04_train_encoders.ipynb   ← retrains EEG/MRI/HPO encoders on augmented data
notebooks/05_fusion_train.ipynb     ← retrains fusion model
notebooks/06_evaluate.ipynb         ← evaluates new model
```

After training completes, the fresh `.pt` files will be committed via the last cell:
```python
!git add checkpoints/ reports/
!git commit -m "retrain: fusion model on augmented MRI dataset"
!git push origin main
```

Then pull on Windows: `git pull origin main` — your local checkpoints are updated.

---

## Docker (Alternative)

If Docker Desktop is installed on Windows, you can skip the conda setup entirely:

```batch
REM Copy checkpoints/ and datasets/ first (or let setup.bat download them)
docker compose up --build
```

Then open http://localhost:8501.
