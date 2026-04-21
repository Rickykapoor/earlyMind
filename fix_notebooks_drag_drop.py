import json
from pathlib import Path

def update_notebook_for_drag_and_drop(path, extract_dir, repo_name, tar_name, ext, label_files="*.csv"):
    with open(path, 'r') as f:
        nb = json.load(f)
    
    # 1. Update Cell 4
    pull_cell = nb['cells'][4]
    
    new_source = [
        f"# CELL 4: Data Loader (Supports Drag-and-Drop OR Automatic Download)\n",
        f"import os, glob, shutil\n",
        f"os.makedirs('{extract_dir}', exist_ok=True)\n",
        f"\n",
        f"# 1. Check if user dragged-and-dropped the raw files directly to /content/\n",
        f"if len(glob.glob('/content/*{ext}')) > 0:\n",
        f"    print('Found raw files in /content/! Moving them to {extract_dir}...')\n",
        f"    os.system(f'mv /content/*{ext} {extract_dir}/')\n",
        f"    os.system(f'mv /content/{label_files} {extract_dir}/ 2>/dev/null')\n",
        f"\n",
        f"# 2. Check if user dragged-and-dropped the .tar.gz file to /content/\n",
        f"elif os.path.exists('/content/{tar_name}'):\n",
        f"    print('Found {tar_name} in /content/! Extracting...')\n",
        f"    os.system(f'tar -xzf /content/{tar_name} -C datasets/{repo_name}')\n",
        f"\n",
        f"# 3. Fallback: Automatically download from GitHub Releases\n",
        f"else:\n",
        f"    print('Downloading dataset automatically from GitHub Releases...')\n",
        f"    os.system(f'wget -qO {tar_name} https://github.com/Rickykapoor/earlyMind/releases/download/v1.0.0-data/{tar_name}')\n",
        f"    os.system(f'tar -xzf {tar_name} -C datasets/{repo_name}')\n",
        f"\n",
        f"print('Datasets ready.')\n"
    ]
        
    pull_cell['source'] = new_source
    pull_cell['cell_type'] = 'code'

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

base = Path('notebooks')
update_notebook_for_drag_and_drop(base / '01_eeg_preprocess.ipynb', 'datasets/eeg/helsinki_neonatal', 'eeg', 'eeg_raw.tar.gz', '.edf', '*.csv')
update_notebook_for_drag_and_drop(base / '02_mri_preprocess.ipynb', 'datasets/mri/baby_open_brains', 'mri', 'mri_raw.tar.gz', '.nii.gz', '*.tsv')
update_notebook_for_drag_and_drop(base / '03_hpo_preprocess.ipynb', 'datasets/facial/hpo', 'facial', 'facial_raw.tar.gz', '.hpoa', '*.hpoa')

print("Updated notebooks successfully for drag-and-drop!")
