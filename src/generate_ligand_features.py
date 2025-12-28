
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

# Step 3: Generate Morgan fingerprints
#   Generate Morgan (ECFP) fingerprints for each ligand

# Step 4: (Optional) Add molecular descriptors
#   Optionally add basic molecular descriptors

# Step 5: Cache to disk
#   Cache ligand features to disk