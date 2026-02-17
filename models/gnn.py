#Converting ligand graphs to 128-dimension vectors

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):

    def __init__(self, inFeatures, outFeatures):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(inFeatures, outFeatures)
        self.edgeLinear = nn.Linear(6, outFeatures)  # 6-dim edge features

    def forward(self, nodeFeatures, edgeIndex, edgeFeatures):
        numNodes = nodeFeatures.size(0)
        transformed = self.linear(nodeFeatures)
        aggregated = torch.zeros(numNodes, transformed.size(1), device=nodeFeatures.device)
        for idx, (src, dst) in enumerate(edgeIndex):
            edgeMessage = self.edgeLinear(edgeFeatures[idx])
            aggregated[dst] += transformed[src] + edgeMessage
        return F.relu(aggregated)
    
class GNNEncoder(nn.Module):

    def __init__(self, nodeFeaturesDimension = 19, hiddenDimension = 128, outputDimension=128, numLayers = 3):
        super(GNNEncoder, self).__init__()
        self.numLayers = numLayers
        self.convLayers = nn.ModuleList()
        self.convLayers.append(GraphConvolutionLayer(nodeFeaturesDimension, hiddenDimension))
        for layer in range(numLayers - 2):
            self.convLayers.append(GraphConvolutionLayer(hiddenDimension, hiddenDimension))
        self.convLayers.append(GraphConvolutionLayer(hiddenDimension, outputDimension))
        self.batchNorms = nn.ModuleList([
            nn.BatchNorm1d(hiddenDimension if i < numLayers-1 else outputDimension)
            for i in range(numLayers)
        ])

    def forward(self, nodeFeatures, edgeIndex, edgeFeatures):
        x = nodeFeatures
        for i, (conv, bn) in enumerate(zip(self.convLayers, self.batchNorms)):
            x = conv(x, edgeIndex, edgeFeatures)
            if x.size(0) > 1:
                x = bn(x)
            if i < self.numLayers-1:
                x = F.dropout(x, p=0.2, training=self.training)
        graphEmbedding = torch.mean(x, dim=0)
        return graphEmbedding
    
class BindingAffinityGNN(nn.Module):

    def __init__(self, proteinDimension=640, ligandGnnOutput=128, hiddenDimension=256):
        super(BindingAffinityGNN, self).__init__()
        self.ligandEncoder = GNNEncoder(
            nodeFeaturesDimension=19,
            hiddenDimension=128,
            outputDimension=ligandGnnOutput,
            numLayers=3
        )
        combinedDimension = proteinDimension + ligandGnnOutput
        self.mlp = nn.Sequential(
            nn.Linear(combinedDimension, hiddenDimension),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hiddenDimension, hiddenDimension // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hiddenDimension // 2, 1)
        )
    
    def forward(self, proteinEmbedding, nodeFeatures, edgeIndex, edgeFeatures):
        ligandEmbedding = self.ligandEncoder(nodeFeatures, edgeIndex, edgeFeatures)
        combined = torch.cat([proteinEmbedding, ligandEmbedding], dim=0)
        prediction = self.mlp(combined)

        return prediction


if __name__ == '__main__':
    import pickle
    import pandas as pd

    # Load ligand graphs
    with open('embeddings/ligands/ligand_graphs.pkl', 'rb') as f:
        ligandGraphs = pickle.load(f)

    # Load protein embeddings
    with open('embeddings/proteins/protein_embeddings.pkl', 'rb') as f:
        proteinEmbeddings = pickle.load(f)

    # Load training data to get actual pKd values
    trainDf = pd.read_csv('data/splits/train.csv', low_memory=False)

    # Get first sample from training data (has protein, ligand, and pKd)
    firstRow = trainDf.iloc[0]
    smiles = firstRow['Ligand SMILES']
    proteinSeq = firstRow['BindingDB Target Chain Sequence 1']
    actualPKd = firstRow['pKd']

    # Get corresponding graph and embedding
    graph = ligandGraphs[smiles]
    proteinEmb = proteinEmbeddings[proteinSeq]

    # Create complete model
    model = BindingAffinityGNN(proteinDimension=480, ligandGnnOutput=128, hiddenDimension=256)

    # Convert to tensors
    proteinTensor = torch.FloatTensor(proteinEmb)
    nodeFeatures = torch.FloatTensor(graph['nodeFeatures'])
    edgeFeatures = torch.FloatTensor(graph['edgeFeatures'])
    edgeIndex = graph['edgeIndex']

    # Forward pass
    prediction = model(proteinTensor, nodeFeatures, edgeIndex, edgeFeatures)