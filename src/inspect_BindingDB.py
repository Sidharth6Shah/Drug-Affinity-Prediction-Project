import pandas as pd
from pathlib import Path
import numpy as np

df_sample = pd.read_csv('data/raw/BindingDB_All.tsv', sep='\t', nrows=5)
# print(df_sample.columns)

#Empty array to store processed chunks in continuously
processed_chunks = []

for chunk in pd.read_csv('data/raw/BindingDB_All.tsv', sep='\t', chunksize=100000):
    
    #Keep entries with Kd value, and convert the Kd value to molar
    chunk = chunk[chunk['Kd (nM)'].notna()]
    chunk['Kd_M'] = pd.to_numeric(chunk['Kd (nM)'], errors='coerce') * 1e-9

    #Remove incoomplete entries
    chunk = chunk[chunk['Ligand SMILES'].notna()]
    chunk = chunk[chunk['BindingDB Target Chain Sequence 1'].notna()]

    #Filter for human entries
    chunk = chunk[chunk['Target Source Organism According to Curator or DataSource'].str.contains('Homo sapiens', na=False)]

    #Find pKd from Kd
    chunk['pKd'] = -np.log10(chunk['Kd_M'])

    #Keep apending the newly processed chunk
    processed_chunks.append(chunk)

df = pd.concat(processed_chunks, ignore_index=True)

#Removal of duplicates
df = df.drop_duplicates(subset=['Ligand SMILES', 'BindingDB Target Chain Sequence 1'])

df.to_csv('data/processed/bindingdb_clean.csv', index=False)