#Quick script to check max protein sequence length (is truncation needed, as ESM-2's max is 1024)

import pandas as pd

# Load the training data
train_df = pd.read_csv('data/splits/train.csv')

# Get all protein sequences
protein_sequences = train_df['BindingDB Target Chain Sequence 1']

# Calculate lengths
protein_lengths = [len(seq) for seq in protein_sequences]

# Statistics
max_length = max(protein_lengths)
min_length = min(protein_lengths)
avg_length = sum(protein_lengths) / len(protein_lengths)

print(f"Protein Sequence Length Statistics:")
print(f"  Minimum length: {min_length} amino acids")
print(f"  Maximum length: {max_length} amino acids")
print(f"  Average length: {avg_length:.2f} amino acids")
print(f"\nProteins > 1024 amino acids: {sum(1 for l in protein_lengths if l > 1024)}")
print(f"Proteins > 512 amino acids: {sum(1 for l in protein_lengths if l > 512)}")
