#training script for gnn_iter3 - simpler architecture
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import json

from gnn_iter3 import BindingAffinityGNN

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
    for batch in tqdm(dataloader, desc="training"):
        batchLoss = 0

        #process each sample in batch
        for sample in batch:
            proteinEmbedding = sample['proteinEmbedding'].to(device)
            nodeFeatures = sample['nodeFeatures'].to(device)
            edgeFeatures = sample['edgeFeatures'].to(device)
            edgeIndex = sample['edgeIndex']
            label = sample['label'].to(device)

            #no gradient
            optimizer.zero_grad()

            #forward pass
            prediction = model(proteinEmbedding, nodeFeatures, edgeIndex, edgeFeatures)
            
            #find loss
            loss = criterion(prediction, label)
            batchLoss += loss.item()

            #backward pass
            loss.backward()

            #update weights
            optimizer.step()
        
        totalLoss += batchLoss/len(batch)

    return totalLoss/len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set to evaluation mode (disables dropout)
    totalLoss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for sample in batch:
                proteinEmbedding = sample['proteinEmbedding'].to(device)
                nodeFeatures = sample['nodeFeatures'].to(device)
                edgeFeatures = sample['edgeFeatures'].to(device)
                edgeIndex = sample['edgeIndex']
                label = sample['label'].to(device)
                
                # Forward pass
                prediction = model(proteinEmbedding, nodeFeatures, edgeIndex, edgeFeatures)
                # Calculate loss
                loss = criterion(prediction, label)
                totalLoss += loss.item()
                # Store predictions and labels for metrics
                predictions.append(prediction.cpu().numpy()[0])
                labels.append(label.cpu().numpy()[0])
    
    # Calculations
    averageLoss = totalLoss / sum(len(batch) for batch in dataloader)
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    r2 = r2_score(labels, predictions)
    
    return averageLoss, rmse, r2, predictions, labels



def main():
    
    with open('embeddings/proteins/protein_embeddings.pkl', 'rb') as f:
        proteinEmbeddings = pickle.load(f)
    
    with open('embeddings/ligands/ligand_graphs.pkl', 'rb') as f:
        ligandGraphs = pickle.load(f)
        
    trainDataset = ProteinLigandDataset('data/splits_stratified/train.csv', proteinEmbeddings, ligandGraphs)
    valDataset = ProteinLigandDataset('data/splits_stratified/val.csv', proteinEmbeddings, ligandGraphs)
    testDataset = ProteinLigandDataset('data/splits_stratified/test.csv', proteinEmbeddings, ligandGraphs)
    
    # Create dataloaders
    batchSize = 128
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, collate_fn=ProteinLigandDataset.collate)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, collate_fn=ProteinLigandDataset.collate)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False, collate_fn=ProteinLigandDataset.collate)
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BindingAffinityGNN(
        proteinDimension=640,
        ligandGnnOutput=128,
        hiddenDimension=256
    ).to(device)
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=0.00004)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    numEpochs = 50 ##CHANGE TO 50#######################################################################################################################################################################
    bestValRmse = float('inf')
    patience = 10
    patienceCounter = 0
        
    for epoch in range(numEpochs):
        
        # Train
        trainLoss = trainEpoch(model, trainLoader, optimizer, criterion, device)
        
        # Validate
        valLoss, valRmse, valR2, _, _ = evaluate(model, valLoader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(valLoss)
        
        # Early stopping
        if valRmse < bestValRmse:
            bestValRmse = valRmse
            patienceCounter = 0
            torch.save(model.state_dict(), 'results/stratified_split/iter3_best_model.pt')
        else:
            patienceCounter += 1
            if patienceCounter >= patience:
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('results/stratified_split/iter3_best_model.pt'))
    
    testLoss, testRmse, testR2, _, _ = evaluate(model, testLoader, criterion, device)
        
    # Save results
    results = {
        'model': 'GNN_iter3',
        'test_metrics': {
            'rmse': float(testRmse),
            'r2': float(testR2)
        }
    }
    
    with open('results/stratified_split/iter3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    

if __name__ == '__main__':
    main()