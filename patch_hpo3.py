import json
from pathlib import Path

# 1. Update Notebook 03
nb_path = Path('notebooks/03_hpo_preprocess.ipynb')
with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        skip_next = False
        for line in cell['source']:
            if "skip = 0" in line:
                skip_next = True
            elif "df_hpo_raw = pd.read_csv" in line and skip_next:
                new_source.append("df_hpo_raw = pd.read_csv(hpoa_path, sep='\\t', skiprows=4, low_memory=False)\n")
                skip_next = False
            elif not skip_next:
                new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

# 2. Update hpo_loader.py
loader_path = Path('src/data/hpo_loader.py')
with open(loader_path, 'r') as f:
    loader_code = f.read()

old_code = """    # Find the header row dynamically
    skip = 0
    with open(phenotype_path, "r") as f:
        for i, line in enumerate(f):
            if 'DatabaseID' in line:
                skip = i
                break

    sep = "\\t"
    df = pd.read_csv(phenotype_path, sep=sep, skiprows=skip, low_memory=False)"""

new_code = """    # Hardcode skip 4 comment lines
    sep = "\\t"
    df = pd.read_csv(phenotype_path, sep=sep, skiprows=4, low_memory=False)"""

loader_code = loader_code.replace(old_code, new_code)
with open(loader_path, 'w') as f:
    f.write(loader_code)

print("HPOA parser successfully patched with hardcoded skiprows=4!")
