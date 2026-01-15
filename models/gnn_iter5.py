#gnn attempt 5 (simplified architecture)
# Improvements from gnn_iter4:
# - gnn returns embeddings per node instead of pooled
# - implement a cross-attention module between gnn and mlp


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

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, proteinDimension=480, ligandDimension=128, numHeads=4):
        super(MultiHeadCrossAttention, self).__init__()
        assert ligandDimension % numHeads == 0, ""
        self.numHeads = numHeads
        self.headDimension = ligandDimension // numHeads
        self.scale = self.headDimension ** 0.5
        self.queryProjection = nn.Linear(proteinDimension, ligandDimension)
        self.keyProjection = nn.Linear(ligandDimension, ligandDimension)
        self.valueProjection = nn.Linear(ligandDimension, ligandDimension)
        self.outputProjection = nn.Linear(ligandDimension, ligandDimension)

    def forward(self, proteinEmbedding, ligandNodes):
        N = ligandNodes.size(0)
        query = self.queryProjection(proteinEmbedding).view(1, self.numHeads, self.headDimension)
        keys = self.keyProjection(ligandNodes).view(N, self.numHeads, self.headDimension)
        values = self.valueProjection(ligandNodes).view(N, self.numHeads, self.headDimension)
        query = query.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)
        scores = torch.matmul(query, keys.transpose(1, 2)) / self.scale
        attentionWeights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attentionWeights, values)
        attended = attended.transpose(0, 1).contiguous().view(1, -1)
        output = self.outputProjection(attended).squeeze(0)
        return output

class GNNEncoder(nn.Module):

    def __init__(self, nodeFeaturesDimension = 19, hiddenDimension = 128, outputDimension=128, numLayers = 3):
        super(GNNEncoder, self).__init__()
        self.numLayers = numLayers
        self.convLayers = nn.ModuleList()
        self.convLayers.append(GraphConvolutionLayer(nodeFeaturesDimension, hiddenDimension))
        self.convLayers.append(GraphConvolutionLayer(hiddenDimension, hiddenDimension))
        self.convLayers.append(GraphConvolutionLayer(hiddenDimension, outputDimension))
        self.layerNorms = nn.ModuleList([
            nn.LayerNorm(hiddenDimension if i < numLayers-1 else outputDimension)
            for i in range(numLayers)
        ])

    def forward(self, nodeFeatures, edgeIndex, edgeFeatures):
        x = nodeFeatures
        for i, (conv, ln) in enumerate(zip(self.convLayers, self.layerNorms)):
            residual = x
            x = conv(x, edgeIndex, edgeFeatures)
            x = ln(x)
            if i > 0 and i < self.numLayers - 1:
                x = x + residual
            if i < self.numLayers - 1:
                x = F.dropout(x, p=0.2, training=self.training)
        return x
    
class BindingAffinityGNN(nn.Module):

    def __init__(self, proteinDimension=480, ligandGnnOutput=128, hiddenDimension=256):
        super(BindingAffinityGNN, self).__init__()
        self.ligandEncoder = GNNEncoder(
            nodeFeaturesDimension=19,
            hiddenDimension=128,
            outputDimension=ligandGnnOutput,
            numLayers=3
        )
        self.crossAttention = MultiHeadCrossAttention(proteinDimension, ligandGnnOutput, numHeads=4)
        combinedDimension = proteinDimension + ligandGnnOutput
        self.mlp = nn.Sequential(
            nn.Linear(combinedDimension, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),


            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1)
        )
    
    def forward(self, proteinEmbedding, nodeFeatures, edgeIndex, edgeFeatures):
        ligandNodes = self.ligandEncoder(nodeFeatures, edgeIndex, edgeFeatures)
        attendedLigand = self.crossAttention(proteinEmbedding, ligandNodes)
        combined = torch.cat([proteinEmbedding, attendedLigand], dim=0)
        prediction = self.mlp(combined)
        return prediction


if __name__ == '__main__':
    import pickle
    import pandas as pd

    print("="*50)
    print("TESTING COMPLETE BINDING AFFINITY MODEL")
    print("="*50)

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

    print(f"\nTest ligand SMILES: {smiles}")
    print(f"Ligand - Nodes: {graph['numNodes']}, Edges: {len(graph['edgeIndex'])}")
    print(f"Protein sequence length: {len(proteinSeq)} amino acids")
    print(f"Protein embedding shape: {proteinEmb.shape}")
    print(f"Actual pKd: {actualPKd:.4f}")

    # Create complete model
    model = BindingAffinityGNN(proteinDimension=480, ligandGnnOutput=128, hiddenDimension=256)

    # Convert to tensors
    proteinTensor = torch.FloatTensor(proteinEmb)
    nodeFeatures = torch.FloatTensor(graph['nodeFeatures'])
    edgeFeatures = torch.FloatTensor(graph['edgeFeatures'])
    edgeIndex = graph['edgeIndex']

    # Forward pass
    prediction = model(proteinTensor, nodeFeatures, edgeIndex, edgeFeatures)

    print(f"\n" + "="*50)
    print(f"Actual pKd:    {actualPKd:.4f}")
    print(f"Predicted pKd: {prediction.item():.4f}")
    print(f"Difference:    {abs(actualPKd - prediction.item()):.4f}")
    print(f"\nOutput shape: {prediction.shape}")
    print(f"Expected shape: torch.Size([1])")
    print(f"\nSuccess! ✓" if prediction.shape[0] == 1 else "Failed ✗")
    print("="*50)




#NOTE this model overfit
# Next steps:
# - 