import json
from pathlib import Path

# 1. Update Notebook 03
nb_path = Path('notebooks/03_hpo_preprocess.ipynb')
with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell['source']:
            if "if l.startswith('DatabaseID') or l.startswith('#DatabaseID'):" in line:
                new_source.append("        if 'DatabaseID' in l:\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

# 2. Update hpo_loader.py
loader_path = Path('src/data/hpo_loader.py')
with open(loader_path, 'r') as f:
    loader_code = f.read()

old_code = "if line.startswith('DatabaseID') or line.startswith('#DatabaseID'):"
new_code = "if 'DatabaseID' in line:"

loader_code = loader_code.replace(old_code, new_code)
with open(loader_path, 'w') as f:
    f.write(loader_code)

print("HPOA parser successfully patched to be BOM/whitespace immune!")
