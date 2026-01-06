import pandas as pd
import numpy as np

#Load all splits to get ALL unique proteins across train/val/test
train_df = pd.read_csv('data/splits/train.csv', low_memory=False)
val_df = pd.read_csv('data/splits/val.csv', low_memory=False)
test_df = pd.read_csv('data/splits/test.csv', low_memory=False)

#Combine all unique proteins from all splits
all_proteins = pd.concat([
    train_df['BindingDB Target Chain Sequence 1'],
    val_df['BindingDB Target Chain Sequence 1'],
    test_df['BindingDB Target Chain Sequence 1']
])
unique_proteins = all_proteins.unique()


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

def getProteinEmbedding(sequence, model, tokenizer, device):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024) #Truncation needed (protein_max_length_check.py)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.last_hidden_state
    embedding = hidden_states.mean(dim=1)
    embedding = embedding.cpu().numpy().squeeze()
    return embedding


from pathlib import Path
from tqdm import tqdm
import pickle

embeddingsDir = Path("embeddings/proteins")
embeddingsDir.mkdir(parents=True, exist_ok=True)
proteinEmbeddings = {}

for sequence in tqdm(unique_proteins, desc="Generating protein embeddings"):
    embedding = getProteinEmbedding(sequence, model, tokenizer, device)
    proteinEmbeddings[sequence] = embedding

with open(embeddingsDir / "protein_embeddings.pkl", "wb") as f:
    pickle.dump(proteinEmbeddings, f)