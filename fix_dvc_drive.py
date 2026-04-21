import json
from pathlib import Path

def update_drive(path):
    with open(path, 'r') as f:
        nb = json.load(f)
    
    cell4 = nb['cells'][4]
    
    new_source = []
    new_source.append("# CELL 4: Data Loader (Supports DVC on Mounted Google Drive, Drag-and-Drop, or Auto Download)\n")
    new_source.append("import os, glob, shutil\n")
    
    # We must preserve the os.makedirs line from earlier edits
    os_makedirs_line = cell4['source'][2]
    new_source.append(os_makedirs_line)
    new_source.append("\n")
    
    new_source.append("# OPTION A: If your DVC cache is in your own Google Drive, point DVC to your mounted drive!\n")
    new_source.append("# => Update this to the exact folder path of your DVC storage inside your Google Drive\n")
    new_source.append("MY_DRIVE_DVC_PATH = '/content/drive/MyDrive/earlyMind_DVC' # <-- Update this!\n")
    new_source.append("\n")
    new_source.append("if os.path.exists(MY_DRIVE_DVC_PATH):\n")
    new_source.append("    print(f'Found local DVC remote at {MY_DRIVE_DVC_PATH}...')\n")
    new_source.append("    os.system(f'dvc remote add -d local_gdrive {MY_DRIVE_DVC_PATH} --force')\n")
    new_source.append("    os.system('dvc pull')\n")
    new_source.append("    print('DVC pull complete from mounted Google Drive!')\n")
    new_source.append("\n")
    
    for line in cell4['source'][4:]:
        new_source.append(line.replace("# 1. Check if", "# OPTION B: Check if"))
        
    cell4['source'] = new_source
    
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

base = Path('notebooks')
update_drive(base / '01_eeg_preprocess.ipynb')
update_drive(base / '02_mri_preprocess.ipynb')
update_drive(base / '03_hpo_preprocess.ipynb')

print("Updated notebooks successfully for DVC Google Drive local mount!")
