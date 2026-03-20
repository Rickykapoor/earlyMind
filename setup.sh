#!/usr/bin/env bash
# setup.sh — One-shot environment setup for EarlyMind
# Run from the project root: bash setup.sh

set -e
echo "======================================================"
echo "  EarlyMind — Environment Setup"
echo "======================================================"

# 1. Verify conda env
CONDA_ENV="infant_id"
PYTHON="/opt/anaconda3/envs/${CONDA_ENV}/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Creating conda environment '${CONDA_ENV}' ..."
    conda env create -f environment.yml
else
    echo "Conda env '${CONDA_ENV}' already exists."
fi

echo ""
echo "Python: $($PYTHON --version)"

# 2. Install pip packages
echo ""
echo "Installing pip packages ..."
"${PYTHON}" -m pip install -q -r requirements.txt

# 3. Verify key imports
echo ""
echo "Verifying imports ..."
"${PYTHON}" -c "
import torch, mne, nibabel, streamlit, plotly, sklearn, dvc
print('  torch:      ', torch.__version__)
print('  MPS:        ', torch.backends.mps.is_available())
print('  mne:        ', mne.__version__)
print('  nibabel:    ', nibabel.__version__)
print('  streamlit:  ', streamlit.__version__)
print('All imports OK.')
"

# 4. Create directory structure
echo ""
echo "Creating directories ..."
"${PYTHON}" -c "
from src.config import cfg
cfg.paths.makedirs()
print('  All output directories created.')
"

# 5. DVC remote
echo ""
echo "Configuring DVC remote ..."
/opt/anaconda3/envs/${CONDA_ENV}/bin/dvc remote add -d gdrive \
    gdrive://19DlpHZ5QCBcFKvfKHprIUhUeVg5bDxU4 --force 2>/dev/null || true

echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. dvc pull            (download datasets)"
echo "  2. Open Colab and run notebooks in order:"
echo "     01_eeg_preprocess.ipynb"
echo "     02_mri_preprocess.ipynb"
echo "     03_hpo_preprocess.ipynb"
echo "     04_train_encoders.ipynb"
echo "     05_fusion_train.ipynb"
echo "     06_evaluate.ipynb"
echo "  3. git pull && dvc pull"
echo "  4. /opt/anaconda3/envs/infant_id/bin/streamlit run app.py"
echo "======================================================"
