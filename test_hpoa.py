import pandas as pd

def read_hpoa(path):
    # Find the header row
    skip = 0
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#DatabaseID') or line.startswith('DatabaseID'):
                skip = i
                break
    print(f"Header found at line {skip}")
    df = pd.read_csv(path, sep='\t', skiprows=skip, low_memory=False)
    print("Shape:", df.shape)
    print(df.head())
    
read_hpoa('datasets/facial/hpo/phenotype.hpoa')
