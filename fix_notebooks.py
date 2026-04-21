import json
from pathlib import Path

def update_notebook(path, dvc_pull_idx, repo_name, tar_name, ext_dir):
    with open(path, 'r') as f:
        nb = json.load(f)
    
    # 1. Update DVC pull cell
    pull_cell = nb['cells'][dvc_pull_idx]
    
    if "01_eeg" in path.name:
        new_source = [
            "# CELL 4: Pull datasets via GitHub Releases (must run BEFORE any imports that load data)\n",
            f"!mkdir -p {ext_dir}\n",
            f"!wget -qO {tar_name} https://github.com/Rickykapoor/earlyMind/releases/download/v1.0.0-data/{tar_name}\n",
            f"!tar -xzf {tar_name} -C datasets/eeg\n",
            "# Verify EDF files are present\n",
            "import glob\n",
            "edf_files = glob.glob('datasets/eeg/helsinki_neonatal/*.edf')\n",
            "print(f'EDF files found: {sorted(edf_files)}')\n",
            "if len(edf_files) == 0:\n",
            "    raise RuntimeError('No EDF files found! Check that dataset download completed successfully.')"
        ]
    else:
        new_source = [
            f"!mkdir -p {ext_dir}\n",
            f"!wget -qO {tar_name} https://github.com/Rickykapoor/earlyMind/releases/download/v1.0.0-data/{tar_name}\n",
            f"!tar -xzf {tar_name} -C datasets/{repo_name}\n",
            "print('Datasets ready.')"
        ]
        
    pull_cell['source'] = new_source
    
    # 2. Update DVC push cell (usually the last cell)
    push_cell = nb['cells'][-1]
    new_push_source = []
    for line in push_cell['source']:
        if "!dvc push" not in line and '"dvc[gdrive]"' not in line:
            new_push_source.append(line)
            
    push_cell['source'] = new_push_source
    
    # Also remove dvc from pip install cell (usually cell 2)
    pip_cell = nb['cells'][2]
    new_pip = []
    for line in pip_cell['source']:
        line = line.replace(' "dvc[gdrive]"', '').replace(" 'dvc[gdrive]'", '')
        new_pip.append(line)
    pip_cell['source'] = new_pip

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

base = Path('notebooks')
update_notebook(base / '01_eeg_preprocess.ipynb', 4, 'eeg', 'eeg_raw.tar.gz', 'datasets/eeg/helsinki_neonatal')
update_notebook(base / '02_mri_preprocess.ipynb', 4, 'mri', 'mri_raw.tar.gz', 'datasets/mri/baby_open_brains')
update_notebook(base / '03_hpo_preprocess.ipynb', 4, 'facial', 'facial_raw.tar.gz', 'datasets/facial/hpo')

print("Updated notebooks successfully!")
