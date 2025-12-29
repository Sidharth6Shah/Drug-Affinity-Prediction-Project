#Phase 5: Feature assembly

import pickle
import pandas as pd
import numpy as np

#Load in the precomputed embeddings
with open('embeddings/proteins/protein_embeddings.pkl', 'rb') as f:
    proteinEmbeddings = pickle.load(f)

#Load in the precomputed fingerprints
with open('embeddings/ligands/ligand_fingerprints.pkl', 'rb') as f:
    ligandFingerprints = pickle.load(f)

#Load in the data splits to identify which samples to process where
train_df = pd.read_csv('data/splits/train.csv', low_memory=False)
val_df = pd.read_csv('data/splits/val.csv', low_memory=False)
test_df = pd.read_csv('data/splits/test.csv', low_memory=False)

#Concatenate features --> protein embedding + ligand fingerprint
def assembleFeatures(df, proteinEmbeddings, ligandFingerprints):
    features = []
    labels = []

    for idx, row in df.iterrows():
        proteinSequence = row['BindingDB Target Chain Sequence 1']
        ligandSMILES = row['Ligand SMILES']
        label = row['pKd']

        #Get the precomputed embeddings/fingerprints
        proteinEmbedding = proteinEmbeddings.get(proteinSequence)
        ligandFingerprint = ligandFingerprints.get(ligandSMILES)

        #Concatenate
        if proteinEmbedding is not None and ligandFingerprint is not None:
            combinedFeature = np.concatenate((proteinEmbedding, ligandFingerprint))
            features.append(combinedFeature)
            labels.append(label)

    X = np.array(features)
    Y = np.array(labels)

    return X, Y

#Create X and Y for each split
X_train, Y_train = assembleFeatures(train_df, proteinEmbeddings, ligandFingerprints)
X_val, Y_val = assembleFeatures(val_df, proteinEmbeddings, ligandFingerprints)
X_test, Y_test = assembleFeatures(test_df, proteinEmbeddings, ligandFingerprints)

np.save('data/final/X_train.npy', X_train)
np.save('data/final/Y_train.npy', Y_train)
np.save('data/final/X_val.npy', X_val)
np.save('data/final/Y_val.npy', Y_val)
np.save('data/final/X_test.npy', X_test)
np.save('data/final/Y_test.npy', Y_test)