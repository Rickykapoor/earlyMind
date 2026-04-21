@echo off
REM =============================================================================
REM  EarlyMind — Windows Setup Script
REM  Run from the project root: setup.bat
REM
REM  Prerequisites:
REM    - Anaconda / Miniconda installed  (https://www.anaconda.com/download)
REM    - Git installed                   (https://git-scm.com/download/win)
REM    - Internet connection (downloads ~1.5 GB of model checkpoints + datasets)
REM =============================================================================

setlocal enabledelayedexpansion

echo.
echo ======================================================
echo   EarlyMind -- Windows Setup
echo ======================================================
echo.

REM ── 0. Sanity-check for conda ────────────────────────────────────────────────
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda not found. Please install Anaconda first:
    echo         https://www.anaconda.com/download
    pause
    exit /b 1
)

where curl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] curl not found. Windows 10/11 ships with curl; please update your OS.
    pause
    exit /b 1
)

where tar >nul 2>&1
if errorlevel 1 (
    echo [ERROR] tar not found. Windows 10 1803+ includes tar. Please update your OS.
    pause
    exit /b 1
)

REM ── 1. Create / verify conda environment ─────────────────────────────────────
echo [1/6] Setting up conda environment 'infant_id' ...
conda info --envs | findstr /C:"infant_id" >nul 2>&1
if errorlevel 1 (
    echo       Creating new environment from environment.yml ...
    conda env create -f environment.yml
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment. Check environment.yml.
        pause
        exit /b 1
    )
) else (
    echo       Environment 'infant_id' already exists. Skipping creation.
)

REM ── 2. Activate environment and install pip packages ─────────────────────────
echo.
echo [2/6] Installing pip packages ...
call conda activate infant_id
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check requirements.txt.
    pause
    exit /b 1
)
echo       Pip packages installed OK.

REM ── 3. Verify key imports ────────────────────────────────────────────────────
echo.
echo [3/6] Verifying Python imports ...
python -c "import torch, mne, nibabel, streamlit, plotly, sklearn; print('  torch:     ', torch.__version__); print('  CUDA:      ', torch.cuda.is_available()); print('  mne:       ', mne.__version__); print('  nibabel:   ', nibabel.__version__); print('  streamlit: ', streamlit.__version__); print('  All imports OK.')"
if errorlevel 1 (
    echo [ERROR] One or more imports failed. Re-run 'pip install -r requirements.txt' manually.
    pause
    exit /b 1
)

REM ── 4. Create project directory structure ────────────────────────────────────
echo.
echo [4/6] Creating output directories ...
python -c "from src.config import cfg; cfg.paths.makedirs(); print('  Directories OK.')"

REM ── 5. Download large files from GitHub Releases ─────────────────────────────
echo.
echo [5/6] Downloading model checkpoints and raw datasets from GitHub Releases ...
echo       (Total ~1.5 GB -- this will take several minutes depending on connection)
echo.

REM ── Edit RELEASE_BASE_URL with your actual GitHub username / repo ─────────────
set RELEASE_BASE_URL=https://github.com/Rickykapoor/earlyMind/releases/download/v1.0.0-data

REM Checkpoints
if not exist checkpoints mkdir checkpoints

if not exist checkpoints\fusion_model.pt (
    echo   Downloading fusion_model.pt ~485 MB ...
    curl -L --retry 3 --progress-bar -o checkpoints\fusion_model.pt %RELEASE_BASE_URL%/fusion_model.pt
) else ( echo   checkpoints\fusion_model.pt already exists. Skipping. )

if not exist checkpoints\hpo_encoder_pretrained.pt (
    echo   Downloading hpo_encoder_pretrained.pt ~437 MB ...
    curl -L --retry 3 --progress-bar -o checkpoints\hpo_encoder_pretrained.pt %RELEASE_BASE_URL%/hpo_encoder_pretrained.pt
) else ( echo   checkpoints\hpo_encoder_pretrained.pt already exists. Skipping. )

if not exist checkpoints\mri_encoder_pretrained.pt (
    echo   Downloading mri_encoder_pretrained.pt ~42 MB ...
    curl -L --retry 3 --progress-bar -o checkpoints\mri_encoder_pretrained.pt %RELEASE_BASE_URL%/mri_encoder_pretrained.pt
) else ( echo   checkpoints\mri_encoder_pretrained.pt already exists. Skipping. )

if not exist checkpoints\eeg_encoder_pretrained.pt (
    echo   Downloading eeg_encoder_pretrained.pt ~3.3 MB ...
    curl -L --retry 3 --progress-bar -o checkpoints\eeg_encoder_pretrained.pt %RELEASE_BASE_URL%/eeg_encoder_pretrained.pt
) else ( echo   checkpoints\eeg_encoder_pretrained.pt already exists. Skipping. )

REM Raw dataset archives
if not exist eeg_raw.tar.gz (
    echo   Downloading eeg_raw.tar.gz ~83 MB ...
    curl -L --retry 3 --progress-bar -o eeg_raw.tar.gz %RELEASE_BASE_URL%/eeg_raw.tar.gz
) else ( echo   eeg_raw.tar.gz already exists. Skipping. )

if not exist mri_raw.tar.gz (
    echo   Downloading mri_raw.tar.gz ~624 MB ...
    curl -L --retry 3 --progress-bar -o mri_raw.tar.gz %RELEASE_BASE_URL%/mri_raw.tar.gz
) else ( echo   mri_raw.tar.gz already exists. Skipping. )

if not exist facial_raw.tar.gz (
    echo   Downloading facial_raw.tar.gz ~5.8 MB ...
    curl -L --retry 3 --progress-bar -o facial_raw.tar.gz %RELEASE_BASE_URL%/facial_raw.tar.gz
) else ( echo   facial_raw.tar.gz already exists. Skipping. )

REM ── 6. Extract dataset archives ───────────────────────────────────────────────
echo.
echo [6/6] Extracting dataset archives ...

if not exist datasets\eeg\helsinki_neonatal (
    mkdir datasets\eeg\helsinki_neonatal
    echo   Extracting eeg_raw.tar.gz ...
    tar -xzf eeg_raw.tar.gz -C datasets\eeg
) else ( echo   datasets\eeg\helsinki_neonatal already extracted. Skipping. )

if not exist datasets\mri\baby_open_brains (
    mkdir datasets\mri\baby_open_brains
    echo   Extracting mri_raw.tar.gz ...
    tar -xzf mri_raw.tar.gz -C datasets\mri
) else ( echo   datasets\mri\baby_open_brains already extracted. Skipping. )

if not exist datasets\facial\hpo (
    mkdir datasets\facial\hpo
    echo   Extracting facial_raw.tar.gz ...
    tar -xzf facial_raw.tar.gz -C datasets\facial
) else ( echo   datasets\facial\hpo already extracted. Skipping. )

echo.
echo ======================================================
echo   Setup complete!
echo.
echo   Run the app:         start_windows.bat
echo   Run tests:           python -m pytest tests\ -v
echo   Run Colab notebooks: Open notebooks\ in Google Colab
echo                        and run in order 01 through 06
echo ======================================================
echo.
pause
