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


def getBondFeatures(bond):
    features = []
    bondTypes = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    for bondType in bondTypes:
        features.append(1 if bond.GetBondType() == bondType else 0)

    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    
    return features


def smilesToGraph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    nodeFeatures = []
    for atom in mol.GetAtoms():
        nodeFeatures.append(getAtomFeatures(atom))
    edgeIndices = []
    edgeFeatures = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edgeIndices.append((i, j))
        edgeIndices.append((j, i))
        bondFeatures = getBondFeatures(bond)
        edgeFeatures.append(bondFeatures)
        edgeFeatures.append(bondFeatures)

    return {
        'nodeFeatures': nodeFeatures,
        'edgeIndex': edgeIndices,
        'edgeFeatures': edgeFeatures,
        'numNodes': len(nodeFeatures),
    }

from tqdm import tqdm
import pickle
from pathlib import Path

ligandGraphs = {}

for smiles in tqdm(uniqueSMILES):
    graph = smilesToGraph(smiles)
    if graph is not None:
        ligandGraphs[smiles] = graph
Path('embeddings/ligands/').mkdir(parents=True, exist_ok=True)
with open('embeddings/ligands/ligand_graphs.pkl', 'wb') as f:
    pickle.dump(ligandGraphs, f)