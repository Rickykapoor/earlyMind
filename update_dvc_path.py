import json
from pathlib import Path

def update_dvc_path(path):
    with open(path, 'r') as f:
        nb = json.load(f)
    
    cell4 = nb['cells'][4]
    
    new_source = []
    for line in cell4['source']:
        if "MY_DRIVE_DVC_PATH =" in line:
            new_source.append("MY_DRIVE_DVC_PATH = '/content/drive/MyDrive/DVC/Earlyminds/earlyMind_DVC' # <-- Updated to your exact folder!\n")
        else:
            new_source.append(line)
            
    cell4['source'] = new_source
    
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

base = Path('notebooks')
for p in base.glob('*.ipynb'):
    if p.name in ('01_eeg_preprocess.ipynb', '02_mri_preprocess.ipynb', '03_hpo_preprocess.ipynb'):
        update_dvc_path(p)

print("Updated MY_DRIVE_DVC_PATH successfully!")
