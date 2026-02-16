**High level overview:**

- Uses BindingDB dataset to train model to predict protein-ligand binding affinities
    - Only human proteins WITH fully filled out entries are used
- The higher the binding affinity, the better chance the drug has of being effective (not guaranteed, but this allows scientists to eliminate many candidates prior to manual in-lab testing)
- Takes in a protein sequence, and a drug molecule's SMILE string (carbon backbone).
- Outputs predicted binding affinity (BA) between them
    - Metric: pKd

**Tech Stack:**

- Core:
    - Python, PyTorch, NumPy, Pandas, RDKit, etc
- Protein Representation:
    - ESM-2 (Meta)
- Molecule Representation:
    - RDKit
- Model:
    - XGBoost for baseline


**How it Works:**

1. Preprocessing:
    - Load data
    - Isolate human proteins with full entries
    - Normalize affinities**
    - Data is split in 2 ways: random split, and stratified split.
        - Stratified splitting groups all ligands paired with a specific protein in a single set (for eg, all 5 ligands paired with protein A will be in the same set to promote generalization.)
2. Protein Featurization w/ ESM-2:
    - Each protein is passed into the PLM to generate a full embedding. (Each amino acid in the protein gets it's own vector embedding). Shape: [1, nAminoAcids, 640]
    - Mean pooling is used to normalize each protein's embedding to a uniform length. Result shape: [1, 640]
3. Ligand Featurization w/ RDKit:
    - Convert each ligand (in SMILES format) into a fingerprint (extremely long vector capturing chemical patterns and other details of the ligand).
    - Convert SMILE format to graph form for GNN input.
        - Features based on atom type, degree, formal charge, OHE-hybridization, aromaticity, hydrogen count, OHE-bond type, conjugation, and ring membership, summing to 25 features per atom
4. Feature Combination:
    - Numerical representation of the interaction between each protein and it's respective ligand from the dataset.
    - Concatenate ligand fingerprint at the end of protein embedding. **POSSIBLE AREA FOR IMPROVEMENT
    - In the GNN/MLP models, the GNN encodes the ligand data into 128 dimension graphs. These graphs are then concatenated at the end of the protein embeddings and fed into the MLP for final predictions.
5. Models:
- Baseline: XGBoost
- GNN:
    - gnn: 4 layer GNN -> 3 layer MLP
    - gann: 3 layer GNN -> mean pooling -> 5 layer MLP
    - iter3: 4 layer GNN with residual connections and LayerNorm -> 3 layer MLP
    - iter4: 3 layer GNN with residual connections -> 3 layer MLP
    - 3 layer GNN -> Multi head cross attention module -> 4 layer MLP

    - Ordered pairs (feature_combination_embedding, binding_affinity)
    - 2 model paths:
        - XGBoost
        - Multilayer Perceptron (FNN)
    - Train/Val split:
    - Play around with hyperparameters
    - Remember to monitor loss, and stop training when loss stops decreasing
    - Loss functions:
        - R^2
        - RMSE