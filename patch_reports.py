import json
from pathlib import Path

def fix_plot_cells(notebook_path):
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            new_src = []
            for line in cell['source']:
                if 'plt.savefig(' in line and 'cfg.paths.reports.mkdir' not in ''.join(new_src):
                    new_src.append("cfg.paths.reports.mkdir(parents=True, exist_ok=True)\n")
                new_src.append(line)
            cell['source'] = new_src
    
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

base = Path('notebooks')
fix_plot_cells(base / '02_mri_preprocess.ipynb')
fix_plot_cells(base / '03_hpo_preprocess.ipynb')
print("Notebook plotting cells patched successfully!")
