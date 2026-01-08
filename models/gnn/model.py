import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from prep import proteinLigandDataset, model

if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import mean_squared_error, r2_score

    print('Loading data and embeddings...')

    # Load datasets
    train_df = pd.read_csv('../data/splits/train.csv', low_memory=False)
    val_df = pd.read_csv('../data/splits/val.csv', low_memory=False)
    test_df = pd.read_csv('../data/splits/test.csv', low_memory=False)

    # Load cached embeddings
    with open('../embeddings/ligands/ligand_graphs.pkl', 'rb') as f:
        ligandGraphs = pickle.load(f)

    with open('../embeddings/proteins/protein_embeddings.pkl', 'rb') as f:
        proteinEmbeddings = pickle.load(f)

    # Filter out rows with missing graphs or embeddings
    print(f'Before filtering: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test')

    train_df = train_df[train_df['Ligand SMILES'].isin(ligandGraphs.keys())]
    train_df = train_df[train_df['BindingDB Target Chain Sequence 1'].isin(proteinEmbeddings.keys())]

    val_df = val_df[val_df['Ligand SMILES'].isin(ligandGraphs.keys())]
    val_df = val_df[val_df['BindingDB Target Chain Sequence 1'].isin(proteinEmbeddings.keys())]

    test_df = test_df[test_df['Ligand SMILES'].isin(ligandGraphs.keys())]
    test_df = test_df[test_df['BindingDB Target Chain Sequence 1'].isin(proteinEmbeddings.keys())]

    print(f'After filtering: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test')

    # Create datasets and dataloaders
    train_dataset = proteinLigandDataset(train_df, ligandGraphs, proteinEmbeddings)
    val_dataset = proteinLigandDataset(val_df, ligandGraphs, proteinEmbeddings)
    test_dataset = proteinLigandDataset(test_df, ligandGraphs, proteinEmbeddings)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f'Created DataLoaders with batch size 32')

    # Get feature dimensions and initialize model
    sample_graph, sample_protein, sample_label = train_dataset[0]
    nodeFeatureDim = sample_graph.x.shape[1]
    proteinEmbeddingDim = sample_protein.shape[1]

    print(f'Node feature dimension: {nodeFeatureDim}')
    print(f'Protein embedding dimension: {proteinEmbeddingDim}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = model(nodeFeatureDim, proteinEmbeddingDim, hiddenDimension=128)
    gnn_model = gnn_model.to(device)

    print(f'Using device: {device}')
    print(f'Model initialized with {sum(p.numel() for p in gnn_model.parameters())} parameters')

    # Training function
    def train_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0

        for batch_graph, batch_protein, batch_label in loader:
            batch_graph = batch_graph.to(device)
            batch_protein = batch_protein.to(device).squeeze(1)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            predictions = model(batch_graph, batch_protein)
            loss = criterion(predictions, batch_label)

            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # Evaluation function
    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_graph, batch_protein, batch_label in loader:
                batch_graph = batch_graph.to(device)
                batch_protein = batch_protein.to(device).squeeze(1)
                batch_label = batch_label.to(device)

                predictions = model(batch_graph, batch_protein)
                loss = criterion(predictions, batch_label)

                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_label.cpu().numpy())

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()

        valid_mask = ~np.isnan(all_preds)
        all_preds = all_preds[valid_mask]
        all_labels = all_labels[valid_mask]

        if len(all_preds) == 0:
            return float('inf'), float('inf'), 0.0

        avg_loss = total_loss / len(loader)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        r2 = r2_score(all_labels, all_preds)

        return avg_loss, rmse, r2

    # Setup training
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    num_epochs = 50
    start_epoch = 0

    # Check for checkpoint
    checkpoint_path = 'gnn_checkpoint.pt'
    if Path(checkpoint_path).exists():
        print('Loading checkpoint...')
        checkpoint = torch.load(checkpoint_path)
        gnn_model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_rmses = checkpoint['val_rmses']
        best_val_rmse = checkpoint['best_val_rmse']
        print(f'Resuming from epoch {start_epoch}')
    else:
        best_val_rmse = float('inf')
        train_losses = []
        val_losses = []
        val_rmses = []
        print('Starting training from scratch')

    print(f'Training for {num_epochs} epochs')
    print('-' * 60)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(gnn_model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse, val_r2 = evaluate(gnn_model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(gnn_model.state_dict(), 'best_gnn_model.pt')

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': gnn_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_rmses': val_rmses,
                'best_val_rmse': best_val_rmse
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f} | [Checkpoint saved]')
        else:
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}')

    print('-' * 60)
    print(f'Training complete! Best validation RMSE: {best_val_rmse:.4f}')