import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv('data/processed/bindingdb_clean.csv', low_memory=False)
familyProteins = defaultdict(list)

for _, row in df.iterrows():
    proteinSequence = row['BindingDB Target Chain Sequence 1']
    targetName = row['Target Name']
    if pd.notna(proteinSequence) and pd.notna(targetName):
        if proteinSequence not in familyProteins[targetName]:
            familyProteins[targetName].append(proteinSequence)

print(f"Found {len(familyProteins)} protein families")

familySizes = {family: len(proteins) for family, proteins in familyProteins.items()}
familiesWith1Protein = sum(1 for size in familySizes.values() if size == 1)
familiesWith2Proteins = sum(1 for size in familySizes.values() if size == 2)
familiesWith3Plus = sum(1 for size in familySizes.values() if size >= 3)

print(f"\nFamily size distribution:")
print(f"  1 protein:  {familiesWith1Protein}")
print(f"  2 proteins: {familiesWith2Proteins}")
print(f"  3+ proteins: {familiesWith3Plus}")

# Separate by size
singleProteinFamilies = {f: proteins for f, proteins in familyProteins.items() if len(proteins) == 1}
twoProteinFamilies = {f: proteins for f, proteins in familyProteins.items() if len(proteins) == 2}
splittableFamilies = {f: proteins for f, proteins in familyProteins.items() if len(proteins) >= 3}

# Assign to splits
trainProteins = set()
valProteins = set()
testProteins = set()

# 1-protein families → train
for family, proteins in singleProteinFamilies.items():
    trainProteins.add(proteins[0])

# 2-protein families → train/test
for family, proteins in twoProteinFamilies.items():
    trainProteins.add(proteins[0])
    testProteins.add(proteins[1])

# 3+ families → 70/15/15
for family, proteins in splittableFamilies.items():
    np.random.seed(42)
    proteinsCopy = proteins.copy()
    np.random.shuffle(proteinsCopy)
    
    n = len(proteinsCopy)
    nTrain = max(1, int(0.7 * n))
    nVal = max(1, int(0.15 * n))
    
    trainProteins.update(proteinsCopy[:nTrain])
    valProteins.update(proteinsCopy[nTrain:nTrain + nVal])
    testProteins.update(proteinsCopy[nTrain + nVal:])

# Create dataframes
trainData = df[df['BindingDB Target Chain Sequence 1'].isin(trainProteins)]
valData = df[df['BindingDB Target Chain Sequence 1'].isin(valProteins)]
testData = df[df['BindingDB Target Chain Sequence 1'].isin(testProteins)]

# Family overlap analysis
trainFamilies = set(trainData['Target Name'].dropna().unique())
testFamilies = set(testData['Target Name'].dropna().unique())
novelTestFamilies = testFamilies - trainFamilies

# Save
from pathlib import Path
splitsDir = Path('data/splits_stratified')
splitsDir.mkdir(parents=True, exist_ok=True)

trainData.to_csv(splitsDir / 'train.csv', index=False)
valData.to_csv(splitsDir / 'val.csv', index=False)
testData.to_csv(splitsDir / 'test.csv', index=False)