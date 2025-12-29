
#Phase 4: Ligand Featurization
#Goal: Represent ligands numerically using cheminformatics


# Step 1: Load data and get unique ligands
#   Load training data and extract unique SMILES strings

import pandas as pd
import numpy as np

train_df = pd.read_csv('data/splits/train.csv', low_memory=False)
uniqueSmiles = train_df['Ligand SMILES'].unique()
#Same as protein embedding generation, some ligads appear multiple times so the extra computation is not needed

# Step 2: Parse SMILES with RDKit
#   Parse SMILES using RDKit

from rdkit import Chem

def parseSMILES(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

# Step 3: Generate Morgan fingerprints
#   Generate Morgan (ECFP) fingerprints for each ligand

from rdkit.Chem import AllChem

def getMFP(smiles, radius = 2, n_bits=2048):
    #look 2 bonds/connections out, standard size 2048 bits
    mol = Chem.MolFromSmiles(smiles)

#    if mol is None:
#        return np.zeros(n_bits)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)

#Convert returned rdkit value to numpy for ml
    arr = np.zeros(n_bits)
    for i in range (n_bits):
        arr[i] = fp[i]
       
    return arr

# Step 4: (Optional) Add molecular descriptors
#   Optionally add basic molecular descriptors

# Step 5: Cache to disk
#   Cache ligand features to disk