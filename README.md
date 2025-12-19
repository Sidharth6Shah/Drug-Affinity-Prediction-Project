**High level overview:**

- Uses BindingDB dataset to train model to predict protein-ligand binding affinities
    - Only human proteins WITH fully filled out entries are used
- The higher the binding affinity, the better chance the drug has of being effective (not guaranteed, but this allows scientists to eliminate many candidates prior to manual in-lab testing)
- Takes in a protein sequence, and a drug molecule's SMILE string (carbon backbone).
- Outputs predicted binding affinity (BA) between them
    - Possible metrics to measure BA include:
        - Kd (Dissociation Constant)
        - Ki (Inhibition Constant)
        - IC50 (Half-maximal Inhibitory Concentration)
        - **UNLIKELY: EC50 (Half-maximal Effective Concentration)


**Tech Stack:**

- Core:
    - Python, PyTorch, NumPy, Pandas, etc
- Protein Representation:
    - ESM-2 (Meta)
- Molecule Representation:
    - RDKit
- Model:
    - XGBoost, Multilayer Perceptron


**How it Works:**

1. Preprocessing:
    - Load data
    - Isolate human proteins with full entries
    - Normalize affinities**
2. Protein Featurization w/ ESM-2:
    - Each protein is passed into the PLM to generate a full embedding. (Each amino acid in the protein gets it's own vector embedding).
    - Mean pooling is used to normalize each protein's embedding to a uniform length.
    - Cache results
3. Ligand Featurization w/ RDKit:
    - Convert each ligand (in SMILe format) into a fingerprint (extremely long vector capturing chemical patterns and other details of the ligand).
4. Feature Combination:
    - Numerical representation of the interaction between each protein and it's respective ligand from the dataset.
    - Concatenate ligand fingerprint at the end of protein embedding. **POSSIBLE AREA FOR IMPROVEMENT
5. Model:
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