#!/usr/bin/env python3
"""
Quick Test Script for GNN Model Iterations

Purpose: Rapidly test different GNN architectures on a small subset of data
         to evaluate if hyperparameter changes are effective before full training.

         gnn_iter3: 4 layer gnn + 3 layer mlp, 0.4 dropout
         gnn_iter4: 3 layer gnn + 3 layer mlp, 0.5 dropout
         gnn_iter5: 3 layer gnn + cross attention + 3 layer mlp, 0.5 dropout

Usage:
    # Test gnn_iter3 (default: 1000 samples, 5 epochs)
    python scripts/quick_test.py --model gnn_iter3

    # Test gnn_iter4
    python scripts/quick_test.py --model gnn_iter4

    # Custom settings: 2000 samples, 3 epochs
    python scripts/quick_test.py --model gnn_iter4 --samples 2000 --epochs 3

What to look for:
    - Epoch 1 Val R² > -0.05: Good sign (better than GANN's -0.068)
    - Val R² improving each epoch: Model is learning
    - Val R² < -0.1 after epoch 3: Hyperparams likely not helping

Expected runtime: 5-10 minutes on CPU
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import importlib
from datetime import datetime

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

class ProteinLigandDataset(Dataset):
    def __init__(self, splitCsv, proteinEmbeddings, ligandGraphs):
        self.df = pd.read_csv(splitCsv)
        self.proteinEmbeddings = proteinEmbeddings
        self.ligandGraphs = ligandGraphs
        validIndices = []
        for index, row in self.df.iterrows():
            proteinSequence = row['BindingDB Target Chain Sequence 1']
            ligandSmiles = row['Ligand SMILES']
            if proteinSequence in proteinEmbeddings and ligandSmiles in ligandGraphs:
                if pd.notna(row['pKd']) and not np.isinf(row['pKd']):
                    validIndices.append(index)
        self.df = self.df.loc[validIndices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        proteinSequence = row['BindingDB Target Chain Sequence 1']
        proteinEmbeddings = self.proteinEmbeddings[proteinSequence]
        ligandSmiles = row['Ligand SMILES']
        graph = self.ligandGraphs[ligandSmiles]
        label = row['pKd']
        return {
            'proteinEmbedding': torch.FloatTensor(proteinEmbeddings),
            'nodeFeatures': torch.FloatTensor(graph['nodeFeatures']),
            'edgeIndex': graph['edgeIndex'],
            'edgeFeatures': torch.FloatTensor(graph['edgeFeatures']),
            'label': torch.FloatTensor([label])
        }

    def collate(batch):
        return batch

def trainEpoch(model, dataloader, optimizer, criterion, device):
    model.train()
    totalLoss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batchLoss = 0
        for sample in batch:
            proteinEmbedding = sample['proteinEmbedding'].to(device)
            nodeFeatures = sample['nodeFeatures'].to(device)
            edgeFeatures = sample['edgeFeatures'].to(device)
            edgeIndex = sample['edgeIndex']
            label = sample['label'].to(device)
            optimizer.zero_grad()
            prediction = model(proteinEmbedding, nodeFeatures, edgeIndex, edgeFeatures)
            loss = criterion(prediction, label)
            batchLoss += loss.item()
            loss.backward()
            optimizer.step()
        totalLoss += batchLoss/len(batch)
    return totalLoss/len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    totalLoss = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            for sample in batch:
                proteinEmbedding = sample['proteinEmbedding'].to(device)
                nodeFeatures = sample['nodeFeatures'].to(device)
                edgeFeatures = sample['edgeFeatures'].to(device)
                edgeIndex = sample['edgeIndex']
                label = sample['label'].to(device)
                prediction = model(proteinEmbedding, nodeFeatures, edgeIndex, edgeFeatures)
                loss = criterion(prediction, label)
                totalLoss += loss.item()
                predictions.append(prediction.cpu().numpy()[0])
                labels.append(label.cpu().numpy()[0])
    averageLoss = totalLoss / sum(len(batch) for batch in dataloader)
    predictions = np.array(predictions)
    labels = np.array(labels)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    r2 = r2_score(labels, predictions)
    return averageLoss, rmse, r2

def main():
    parser = argparse.ArgumentParser(description='Quick test GNN models on small dataset')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., gnn_iter3, gnn_iter4)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of training samples (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    args = parser.parse_args()

    # Dynamic import
    try:
        model_module = importlib.import_module(args.model)
        BindingAffinityGNN = model_module.BindingAffinityGNN
    except ImportError:
        print(f"Error: Could not import model '{args.model}'")
        print(f"Make sure models/{args.model}.py exists")
        sys.exit(1)

    print("\n" + "="*70)
    print(f"⚡ QUICK TEST MODE")
    print(f"Model: {args.model}")
    print(f"Samples: {args.samples} train, {args.samples//3} val")
    print(f"Epochs: {args.epochs}")
    print("="*70 + "\n")

    # Load data
    with open('embeddings/proteins/protein_embeddings.pkl', 'rb') as f:
        proteinEmbeddings = pickle.load(f)
    with open('embeddings/ligands/ligand_graphs.pkl', 'rb') as f:
        ligandGraphs = pickle.load(f)

    trainDataset = ProteinLigandDataset('data/splits/train.csv', proteinEmbeddings, ligandGraphs)
    valDataset = ProteinLigandDataset('data/splits/val.csv', proteinEmbeddings, ligandGraphs)

    # Limit samples
    trainDataset.df = trainDataset.df.iloc[:args.samples].reset_index(drop=True)
    valDataset.df = valDataset.df.iloc[:args.samples//3].reset_index(drop=True)

    print(f"Loaded {len(trainDataset)} train samples, {len(valDataset)} val samples\n")

    batchSize = 256 #Original is 32
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, collate_fn=ProteinLigandDataset.collate)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, collate_fn=ProteinLigandDataset.collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BindingAffinityGNN(proteinDimension=480, ligandGnnOutput=128, hiddenDimension=256).to(device)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.MSELoss()
    learningRate = 0.0007 #0.0008
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    # Setup logging
    log_dir = 'results'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'quick_test_results.txt')

    # Write header to log file
    with open(log_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Batch Size: {batchSize}\n")
        f.write(f"Learning Rate: {learningRate}\n")
        f.write(f"Train Samples: {len(trainDataset)}\n")
        f.write(f"Val Samples: {len(valDataset)}\n")
        f.write(f"Device: {device}\n")
        f.write("-"*80 + "\n")

    print("="*70)
    print("Training started...")
    print("="*70)
    print(f"Batch size: {batchSize}")
    print(f"Learning rate: {learningRate}")

    for epoch in range(args.epochs):
        trainLoss = trainEpoch(model, trainLoader, optimizer, criterion, device)
        valLoss, valRmse, valR2 = evaluate(model, valLoader, criterion, device)

        status = ""
        if valR2 > 0:
            status = "✓ POSITIVE R²"
        elif valR2 > -0.05:
            status = "⚠ Close to positive"
        else:
            status = "✗ Poor performance"

        # Print to console
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={trainLoss:.4f} | Val R²={valR2:.4f} | Val RMSE={valRmse:.4f} | {status}")

        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.epochs}:\n")
            f.write(f"  Train Loss: {trainLoss:.4f}\n")
            f.write(f"  Val Loss: {valLoss:.4f}\n")
            f.write(f"  Val RMSE: {valRmse:.4f}\n")
            f.write(f"  Val R²: {valR2:.4f}\n")
            f.write(f"  Status: {status}\n")

    # Final summary
    interpretation = ""
    if valR2 > 0.05:
        interpretation = "✓ Good: Positive R² suggests this architecture is promising"
    elif valR2 > -0.05:
        interpretation = "⚠ Marginal: Close to positive, may improve with full training"
    else:
        interpretation = "✗ Poor: Negative R² suggests architecture needs more tuning"

    print("="*70)
    print("\nQuick test complete!")
    print("\nInterpretation:")
    print(interpretation)
    print("="*70 + "\n")

    # Write final summary to log
    with open(log_file, 'a') as f:
        f.write("-"*80 + "\n")
        f.write(f"Final Result: {interpretation}\n")
        f.write("="*80 + "\n\n")

if __name__ == '__main__':
    main()
