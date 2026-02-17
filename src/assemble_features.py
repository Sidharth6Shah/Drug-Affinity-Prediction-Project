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
train_df = pd.read_csv('data/splits_stratified/train.csv', low_memory=False)
val_df = pd.read_csv('data/splits_stratified/val.csv', low_memory=False)
test_df = pd.read_csv('data/splits_stratified/test.csv', low_memory=False)

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
            #Skip samples with invalid labels (NaN or Inf)
            if pd.notna(label) and not np.isinf(label):
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

from pathlib import Path
Path('data/final_stratified').mkdir(parents=True, exist_ok=True)
np.save('data/final_stratified/X_train.npy', X_train)
np.save('data/final_stratified/Y_train.npy', Y_train)
np.save('data/final_stratified/X_val.npy', X_val)
np.save('data/final_stratified/Y_val.npy', Y_val)
np.save('data/final_stratified/X_test.npy', X_test)
np.save('data/final_stratified/Y_test.npy', Y_test)

# print(f"\nFeature assembly complete!")
# print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features (640 protein + 2048 fingerprint)")
# print(f"Val:   {X_val.shape[0]} samples, {X_val.shape[1]} features")
# print(f"Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
# print(f"\nSaved to data/final_stratified/")