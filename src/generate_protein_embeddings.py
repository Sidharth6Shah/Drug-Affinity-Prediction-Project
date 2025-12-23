import pandas as pd
import numpy as np

train_df = pd.read_csv('data/train_proteins.csv')

#Some proteins can be associated with multiple different ligands, so this cuts down on extra embedding generation for repeated proteins
unique_proteins = train_df['protein_sequence'].unique()



from transformers import AutoTokenizer, AutoModel
import torch

#Medium, heavier weight model alternative: "facebook/esm2_t30_150M_UR50D"
modelVariation = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(modelVariation)
model = AutoModel.from_pretrained(modelVariation)

#miniscript to use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()