#Phase 4: Ligand Featurization

# Step 1: Load data and get unique ligands
#   Load training data and extract unique SMILES strings

import pandas as pd
import numpy as np

train_df = pd.read_csv('data/splits/train.csv', low_memory=False)
uniqueSmiles = train_df['Ligand SMILES'].unique()
#Same as protein embedding generation, some ligads appear multiple times so the extra computation is not needed


# Step 2: Generate Morgan (ECFP) fingerprints
#1. convert SMILES to molecular object
#2. generate fingerprint
#3. convert to numpy array

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

# Step 3 (HOLDING): Add molecular descriptors



# Step 4: Caching

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

print(f"\nSaved {len(ligandFingerprints)} ligand fingerprints to {embeddingsDir / 'ligand_fingerprints.pkl'}")