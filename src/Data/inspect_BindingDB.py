import pandas as pd
from pathlib import Path
import numpy as np

df_sample = pd.read_csv('data/raw/BindingDB_All.tsv', sep='\t', nrows=5)
print(df_sample.columns)

for chunk in pd.read_csv('data/raw/BindingDB_All.tsv', sep='\t', chunksize=100000):
    
    chunk = chunk[chunk['Kd (nM)'].notna()]
    chunk['Kd_M'] = chunk['Kd (nM)'] * 1e-9

    chunk = chunk[chunk['Ligand SMILES'].notna()]
    chunk = chunk[chunk['BindingDB Target Chain Sequence'].notna()]
    chunk = chunk[chunk['Target Source Organism...'].str.contains('Homo sapiens', na=False)]

    df = df.drop_duplicates(subset=['Ligand SMILES', 'BindingDB Target Chain Sequence'])

    chunk['pKd'] = -np.log10(chunk['Kd_M'])
    
    df.to_csv('Data/Processed/bindingdb_clean.csv', index=False)