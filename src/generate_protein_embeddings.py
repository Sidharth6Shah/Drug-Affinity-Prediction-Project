import pandas as pd
import numpy as np

train_df = pd.read_csv('data/train_proteins.csv')

#Some proteins can be associated with multiple different ligands, so this cuts down on extra embedding generation for repeated proteins
unique_proteins = train_df['protein_sequence'].unique()

