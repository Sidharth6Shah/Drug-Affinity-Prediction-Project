import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv('data/processed/bindingdb_clean.csv', low_memory=False)
uniqueProteins = df['BindingDB Target Chain Sequence 1'].dropna().unique()

np.random.shuffle(uniqueProteins)
n = len(uniqueProteins)
nTrain = int(0.7 * n)
nVal = int(0.15 * n)

trainProteins = set(uniqueProteins[:nTrain])
valProteins = set(uniqueProteins[nTrain:nTrain + nVal])
testProteins = set(uniqueProteins[nTrain + nVal:])

trainData = df[df['BindingDB Target Chain Sequence 1'].isin(trainProteins)]
valData = df[df['BindingDB Target Chain Sequence 1'].isin(valProteins)]
testData = df[df['BindingDB Target Chain Sequence 1'].isin(testProteins)]

from pathlib import Path
Path('data/splits_stratified').mkdir(exist_ok=True)
trainData.to_csv('data/splits_stratified/train.csv', index=False)
valData.to_csv('data/splits_stratified/val.csv', index=False)
testData.to_csv('data/splits_stratified/test.csv', index=False)

# print(f"\nSplit sizes:")
# print(f"  Train: {len(trainData)} samples, {len(trainProteins)} unique proteins")
# print(f"  Val:   {len(valData)} samples, {len(valProteins)} unique proteins")
# print(f"  Test:  {len(testData)} samples, {len(testProteins)} unique proteins")
# print(f"  Total: {len(trainData) + len(valData) + len(testData)} samples, {n} unique proteins")

# print(f"\nProtein distribution:")
# print(f"  Train: {len(trainProteins)/n*100:.1f}%")
# print(f"  Val:   {len(valProteins)/n*100:.1f}%")
# print(f"  Test:  {len(testProteins)/n*100:.1f}%")