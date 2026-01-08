import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


#model architceture

class model(nn.Module):
    def __init__(self, nodeFeatureDimension, proteinEmbeddingDimension, hiddenDimension = 128, gnnLayers = 3, outputDimension = 1):
        super(model, self).__init__()

        #Layers
        self.conv1 = GCNConv(nodeFeatureDimension, hiddenDimension)
        self.conv2 = GCNConv(hiddenDimension, hiddenDimension)
        self.conv3 = GCNConv(hiddenDimension, hiddenDimension)

        #Multilayer Perceptron for prediction (combining gnn output with protein embeddings)
        combinedDimension = hiddenDimension + proteinEmbeddingDimension
        self.fc1 = nn.Linear(combinedDimension, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, outputDimension)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, graphData, proteinEmbedding):
        x = graphData.x
        edgeIndex = graphData.edge_index
        batch = graphData.batch

        #gnn layers w/ ReLu activation
        x = self.conv1(x, edgeIndex)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edgeIndex)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edgeIndex)
        x = F.relu(x)

        #global pooling
        x = global_mean_pool(x, batch)

        #concatenating gnn embedding w/ protein embedding
        combined = torch.cat([x, proteinEmbedding], dim=1)

        #multilayer perceptron for actual prediction
        x = self.fc1(combined)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x

##dataset class

class proteinLigandDataset(Dataset):
    def __init__(self, df, ligandGraphs, proteinEmbeddings):
        super(proteinLigandDataset, self).__init__()
        self.df = df
        self.ligandGraphs = ligandGraphs
        self.proteinEmbeddings = proteinEmbeddings

    def len(self):
        return len(self.df)

    def get(self, index):
        row = self.df.iloc[index]

        #SMILES and protein sequences
        SMILES = row['Ligand SMILES']
        proteinSequence = row['BindingDB Target Chain Sequence 1']
        label = row['pKd']

        #pull cached embeddings
        graphDictionary = self.ligandGraphs[SMILES]
        proteinEmb = self.proteinEmbeddings[proteinSequence]

        #convert graph dictionaries to pytorch geometric data objects
        nodeFeatures = torch.tensor(graphDictionary['nodeFeatures'], dtype=torch.float)
        edgeIndex = torch.tensor(graphDictionary['edgeIndex'], dtype=torch.long).t().contiguous()

        #create data object
        graphData = Data(x = nodeFeatures, edge_index = edgeIndex)

        #convert protein embeddings to tensors
        proteinTensor = torch.tensor(proteinEmb, dtype=torch.float).unsqueeze(0)
        labelTensor = torch.tensor([label], dtype=torch.float)

        return graphData, proteinTensor, labelTensor