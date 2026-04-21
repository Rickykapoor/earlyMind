import json
import re
from pathlib import Path

# 1. Update Notebook 03
nb_path = Path('notebooks/03_hpo_preprocess.ipynb')
with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell['source']:
            if "pd.read_csv(hpoa_path, sep='\\t')" in line:
                new_source.append("skip = 0\n")
                new_source.append("with open(hpoa_path, 'r') as f:\n")
                new_source.append("    for i, l in enumerate(f):\n")
                new_source.append("        if l.startswith('DatabaseID') or l.startswith('#DatabaseID'):\n")
                new_source.append("            skip = i\n")
                new_source.append("            break\n")
                new_source.append("df_hpo_raw = pd.read_csv(hpoa_path, sep='\\t', skiprows=skip, low_memory=False)\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

# 2. Update hpo_loader.py
loader_path = Path('src/data/hpo_loader.py')
with open(loader_path, 'r') as f:
    loader_code = f.read()

old_code = """    # File uses '#' character at start of header line
    with open(phenotype_path, "r") as f:
        first_line = f.readline()

    sep = "\\t"
    df = pd.read_csv(phenotype_path, sep=sep, comment=None, low_memory=False)

    # Handle comment character in header
    if df.columns[0].startswith("#"):
        df.columns = [c.lstrip("#").strip() for c in df.columns]"""

new_code = """    # Find the header row dynamically
    skip = 0
    with open(phenotype_path, "r") as f:
        for i, line in enumerate(f):
            if line.startswith('DatabaseID') or line.startswith('#DatabaseID'):
                skip = i
                break

    sep = "\\t"
    df = pd.read_csv(phenotype_path, sep=sep, skiprows=skip, low_memory=False)

    # Handle comment character in header
    if df.columns[0].startswith("#"):
        df.columns = [c.lstrip("#").strip() for c in df.columns]"""

loader_code = loader_code.replace(old_code, new_code)
with open(loader_path, 'w') as f:
    f.write(loader_code)

print("HPOA parser successfully patched in Notebook 03 and hpo_loader.py!")
