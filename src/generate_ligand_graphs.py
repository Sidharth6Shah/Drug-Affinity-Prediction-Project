#Phase 7: Ligand Graph Generation for GNN
import pandas as pd
from rdkit import Chem

train_df = pd.read_csv('data/splits/train.csv', low_memory=False)
val_df = pd.read_csv('data/splits/val.csv', low_memory=False)
test_df = pd.read_csv('data/splits/test.csv', low_memory=False)
allSMILES = pd.concat([
    train_df['Ligand SMILES'],
    val_df['Ligand SMILES'],
    test_df['Ligand SMILES']
])
uniqueSMILES = allSMILES.unique()

smiles = "CCO"
mol = Chem.MolFromSmiles(smiles)

def getAtomFeatures(atom):
    features = []
    atomTypes = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atomSymbol = atom.GetSymbol()
    for atomType in atomTypes:
        features.append(1 if atom.GetSymbol() == atomType else 0)
    features.append(1 if atomSymbol not in atomTypes else 0)

    features.append(atom.GetDegree())

    features.append(atom.GetFormalCharge())
    
    hybridizationTypes = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hyb = atom.GetHybridization()
    for hybType in hybridizationTypes:
        features.append(1 if hyb == hybType else 0)
    
    features.append(1 if atom.GetIsAromatic() else 0)
    
    features.append(atom.GetTotalNumHs())
    
    return features

