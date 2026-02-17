#Phase 4: Ligand Featurization

import pandas as pd
import numpy as np

#Load all splits to get ALL unique ligands across train/val/test
train_df = pd.read_csv('data/splits/train.csv', low_memory=False)
val_df = pd.read_csv('data/splits/val.csv', low_memory=False)
test_df = pd.read_csv('data/splits/test.csv', low_memory=False)

#Combine all unique SMILES from all splits
all_smiles = pd.concat([
    train_df['Ligand SMILES'],
    val_df['Ligand SMILES'],
    test_df['Ligand SMILES']
])
uniqueSmiles = all_smiles.unique()

from rdkit import Chem
from rdkit.Chem import AllChem

#look 2 bonds/connections out, standard size 2048 bits
def getMFP(smiles, radius = 2, n_bits=2048):

    #converting the SMILES to a molecular object
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(n_bits)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)

    #Convert returned rdkit value to numpy for ml
    arr = np.zeros(n_bits)
    for i in range (n_bits):
        arr[i] = fp[i]
       
    return arr

from pathlib import Path
from tqdm import tqdm
import pickle

embeddingsDir = Path("embeddings/ligands")
embeddingsDir.mkdir(parents=True, exist_ok=True)
ligandFingerprints = {}

for smiles in tqdm(uniqueSmiles, desc="Generating ligand fingerprints"):
    fp = getMFP(smiles, radius=2, n_bits=2048)
    ligandFingerprints[smiles] = fp

with open(embeddingsDir / "ligand_fingerprints.pkl", "wb") as f:
    pickle.dump(ligandFingerprints, f)

# print(f"\nSaved {len(ligandFingerprints)} ligand fingerprints to {embeddingsDir / 'ligand_fingerprints.pkl'}")