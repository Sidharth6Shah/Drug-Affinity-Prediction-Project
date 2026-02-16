"""
Visualization Generation Script
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
RESULTS_DIR = Path('results/stratified_split')
VIZ_DIR = Path('results/visualizations')
DATA_DIR = Path('data')

# Model configurations
MODELS = {
    'xgboost': {
        'name': 'XGBoost',
        'results_file': 'xgboost_results.json',
        'model_file': 'xgboost_model.pkl',
        'color': '#e74c3c',
        'type': 'baseline'
    },
    'gnn': {
        'name': 'GNN',
        'results_file': 'gnn_results.json',
        'model_file': 'gnn_best_model.pt',
        'color': '#3498db',
        'type': 'gnn'
    },
    'gann': {
        'name': 'GANN (Attention)',
        'results_file': 'gann_results.json',
        'model_file': 'gann_best_model.pt',
        'color': '#9b59b6',
        'type': 'gnn'
    },
    'gnn_iter3': {
        'name': 'GNN Iter3',
        'results_file': 'iter3_results.json',
        'model_file': 'iter3_best_model.pt',
        'color': '#2ecc71',
        'type': 'gnn'
    },
    'gnn_iter4': {
        'name': 'GNN Iter4',
        'results_file': 'iter4_results.json',
        'model_file': 'iter4_best_model.pt',
        'color': '#f39c12',
        'type': 'gnn'
    },
    'gnn_iter5': {
        'name': 'GNN Iter5',
        'results_file': 'iter5_results.json',
        'model_file': 'iter5_best_model.pt',
        'color': '#1abc9c',
        'type': 'gnn'
    }
}


def load_results(model_key):
    results_path = RESULTS_DIR / MODELS[model_key]['results_file']
    if not results_path.exists():
        print(f"Warning: Results file not found for {model_key}")
        return None

    with open(results_path, 'r') as f:
        return json.load(f)


def load_predictions(model_key):
    # Load test data
    X_test = np.load(DATA_DIR / 'final/X_test.npy')
    Y_test = np.load(DATA_DIR / 'final/Y_test.npy')

    model_type = MODELS[model_key]['type']
    model_path = RESULTS_DIR / MODELS[model_key]['model_file']

    if not model_path.exists():
        print(f"Warning: Model file not found for {model_key}")
        return Y_test, None

    if model_type == 'baseline':
        # XGBoost model
        import xgboost as xgb
        model = pickle.load(open(model_path, 'rb'))
        predictions = model.predict(X_test)

    elif model_type == 'gnn':
        # GNN models - need to load graph data and run inference
        predictions = load_gnn_predictions(model_key, Y_test)

    return Y_test, predictions


def load_gnn_predictions(model_key, Y_test):
    print(f"Generating predictions for {model_key}...")

    if model_key == 'gnn':
        from models.gnn import BindingAffinityGNN
    elif model_key == 'gann':
        from models.gann import BindingAffinityGNN
    elif model_key == 'gnn_iter3':
        from models.gnn_iter3 import BindingAffinityGNN
    elif model_key == 'gnn_iter4':
        from models.gnn_iter4 import BindingAffinityGNN
    elif model_key == 'gnn_iter5':
        from models.gnn_iter5 import BindingAffinityGNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BindingAffinityGNN(proteinDimension=480, ligandGnnOutput=128)
    model_path = RESULTS_DIR / MODELS[model_key]['model_file']
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with open('embeddings/proteins/protein_embeddings.pkl', 'rb') as f:
        protein_embeddings = pickle.load(f)

    with open('embeddings/ligands/ligand_graphs.pkl', 'rb') as f:
        ligand_graphs = pickle.load(f)

    test_df = pd.read_csv('data/splits/test.csv', low_memory=False)

    predictions = []

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            smiles = row['Ligand SMILES']
            protein_seq = row['BindingDB Target Chain Sequence 1']

            # Get embeddings
            protein_emb = protein_embeddings[protein_seq]
            graph = ligand_graphs[smiles]

            # Convert to tensors
            protein_tensor = torch.FloatTensor(protein_emb).to(device)
            node_features = torch.FloatTensor(graph['nodeFeatures']).to(device)
            edge_features = torch.FloatTensor(graph['edgeFeatures']).to(device)
            edge_index = graph['edgeIndex']

            # Predict
            pred = model(protein_tensor, node_features, edge_index, edge_features)
            predictions.append(pred.cpu().item())

    return np.array(predictions)


def plot_model_comparison():
    """Create bar charts comparing all models."""
    print("Generating model comparison plots...")

    models_data = []
    for key in MODELS.keys():
        results = load_results(key)
        if results and 'test_metrics' in results:
            models_data.append({
                'name': MODELS[key]['name'],
                'rmse': results['test_metrics']['rmse'],
                'r2': results['test_metrics']['r2'],
                'color': MODELS[key]['color']
            })

    if not models_data:
        print("No model data available for comparison")
        return

    df = pd.DataFrame(models_data)

    # RMSE comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['name'], df['rmse'], color=df['color'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison - RMSE (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'comparisons/rmse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # R² comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['name'], df['r2'], color=df['color'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison - R² (Higher is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(df['r2']) * 1.2])
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'comparisons/r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Combined comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # RMSE
    bars1 = ax1.bar(df['name'], df['rmse'], color=df['color'], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # R²
    bars2 = ax2.bar(df['name'], df['r2'], color=df['color'], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_title('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, max(df['r2']) * 1.2])
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'comparisons/combined_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved comparison plots to {VIZ_DIR / 'comparisons'}")


def plot_predictions_vs_actual(model_key):
    y_true, y_pred = load_predictions(model_key)

    if y_pred is None:
        print(f"  Skipping predictions plot for {model_key} - no predictions available")
        return

    model_name = MODELS[model_key]['name']
    color = MODELS[model_key]['color']

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=color, edgecolors='black', linewidth=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual pKd', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted pKd', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Predicted vs Actual Binding Affinity', fontsize=14, fontweight='bold')

    textstr = f'RMSE: {rmse:.3f}\nR²: {r2:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')

    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = VIZ_DIR / model_key / 'predicted_vs_actual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {model_name} predictions plot")


def plot_residuals(model_key):
    y_true, y_pred = load_predictions(model_key)

    if y_pred is None:
        print(f"  Skipping residuals plot for {model_key} - no predictions available")
        return

    model_name = MODELS[model_key]['name']
    color = MODELS[model_key]['color']

    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residuals vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color=color, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted pKd', fontsize=10, fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=10, fontweight='bold')
    ax.set_title('Residuals vs Predicted Values', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # 2. Histogram of residuals
    ax = axes[0, 1]
    ax.hist(residuals, bins=50, color=color, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residuals', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('Distribution of Residuals', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    textstr = f'Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    # 3. Residuals vs Actual
    ax = axes[1, 0]
    ax.scatter(y_true, residuals, alpha=0.5, s=20, color=color, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Actual pKd', fontsize=10, fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=10, fontweight='bold')
    ax.set_title('Residuals vs Actual Values', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # 4. Absolute residuals vs Predicted
    ax = axes[1, 1]
    ax.scatter(y_pred, np.abs(residuals), alpha=0.5, s=20, color=color, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Predicted pKd', fontsize=10, fontweight='bold')
    ax.set_ylabel('Absolute Residuals', fontsize=10, fontweight='bold')
    ax.set_title('Absolute Error vs Predicted Values', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.suptitle(f'{model_name}: Residual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = VIZ_DIR / model_key / 'residual_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {model_name} residual analysis")


def plot_data_distribution():
    print("Generating data distribution plots...")

    train_df = pd.read_csv('data/splits/train.csv', low_memory=False)
    val_df = pd.read_csv('data/splits/val.csv', low_memory=False)
    test_df = pd.read_csv('data/splits/test.csv', low_memory=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Combined histogram
    ax = axes[0, 0]
    ax.hist(train_df['pKd'], bins=50, alpha=0.6, label='Train', color='#3498db', edgecolor='black')
    ax.hist(val_df['pKd'], bins=50, alpha=0.6, label='Validation', color='#e74c3c', edgecolor='black')
    ax.hist(test_df['pKd'], bins=50, alpha=0.6, label='Test', color='#2ecc71', edgecolor='black')
    ax.set_xlabel('pKd (Binding Affinity)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('pKd Distribution Across Splits', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Box plot
    ax = axes[0, 1]
    data_to_plot = [train_df['pKd'], val_df['pKd'], test_df['pKd']]
    bp = ax.boxplot(data_to_plot, labels=['Train', 'Validation', 'Test'],
                     patch_artist=True, showmeans=True)
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('pKd', fontsize=10, fontweight='bold')
    ax.set_title('pKd Distribution Box Plot', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # 3. Split sizes
    ax = axes[1, 0]
    splits = ['Train', 'Validation', 'Test']
    sizes = [len(train_df), len(val_df), len(test_df)]
    bars = ax.bar(splits, sizes, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Number of Samples', fontsize=10, fontweight='bold')
    ax.set_title('Dataset Split Sizes', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')

    # 4. Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    stats_data = []
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        stats_data.append([
            name,
            f"{df['pKd'].mean():.3f}",
            f"{df['pKd'].std():.3f}",
            f"{df['pKd'].min():.3f}",
            f"{df['pKd'].max():.3f}"
        ])
    table = ax.table(cellText=stats_data,
                     colLabels=['Split', 'Mean', 'Std', 'Min', 'Max'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, 4):
        for j in range(5):
            table[(i, j)].set_facecolor(['#ecf0f1', '#d5dbdb'][i % 2])

    ax.set_title('pKd Statistics by Split', fontsize=11, fontweight='bold', pad=20)

    plt.suptitle('Data Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = VIZ_DIR / 'data_analysis/pKd_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved data distribution plots")


def extract_ligand_embeddings(model_key):
    print(f"  Extracting embeddings for {MODELS[model_key]['name']}...")

    if model_key == 'gnn':
        from models.gnn import BindingAffinityGNN
    elif model_key == 'gann':
        from models.gann import BindingAffinityGNN
    elif model_key == 'gnn_iter3':
        from models.gnn_iter3 import BindingAffinityGNN
    elif model_key == 'gnn_iter4':
        from models.gnn_iter4 import BindingAffinityGNN
    elif model_key == 'gnn_iter5':
        from models.gnn_iter5 import BindingAffinityGNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BindingAffinityGNN(proteinDimension=480, ligandGnnOutput=128)
    model_path = RESULTS_DIR / MODELS[model_key]['model_file']

    if not model_path.exists():
        print(f"    Model file not found, skipping...")
        return None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_df = pd.read_csv('data/splits/test.csv', low_memory=False)
    with open('embeddings/ligands/ligand_graphs.pkl', 'rb') as f:
        ligand_graphs = pickle.load(f)

    embeddings = []
    pkd_values = []

    sample_size = min(2000, len(test_df))
    test_sample = test_df.sample(n=sample_size, random_state=42)

    with torch.no_grad():
        for idx, row in test_sample.iterrows():
            smiles = row['Ligand SMILES']
            graph = ligand_graphs[smiles]

            node_features = torch.FloatTensor(graph['nodeFeatures']).to(device)
            edge_features = torch.FloatTensor(graph['edgeFeatures']).to(device)
            edge_index = graph['edgeIndex']

            embedding = model.ligandEncoder(node_features, edge_index, edge_features)
            embeddings.append(embedding.cpu().numpy())
            pkd_values.append(row['pKd'])

    return np.array(embeddings), np.array(pkd_values)


def plot_ligand_embeddings(model_key):
    if MODELS[model_key]['type'] != 'gnn':
        return

    embeddings, pkd_values = extract_ligand_embeddings(model_key)

    if embeddings is None:
        print(f"  Skipping embedding visualization for {model_key}")
        return

    model_name = MODELS[model_key]['name']

    print(f"  Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=pkd_values, cmap='viridis', s=20, alpha=0.6,
                        edgecolors='black', linewidth=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('pKd (Binding Affinity)', fontsize=11, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}: Learned Ligand Embedding Space\n(colored by binding affinity)',
                fontsize=13, fontweight='bold')

    explanation = ("Each point represents a drug molecule.\n"
                  "Similar drugs cluster together.\n"
                  "Color indicates binding strength (higher = stronger).")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    output_path = VIZ_DIR / model_key / 'ligand_embeddings_tsne.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {model_name} embedding visualization")


def main():
    print("="*60)
    print("GENERATING VISUALIZATIONS FOR BLOG POST")
    print("="*60)

    # 1. Model comparison
    print("\n[1/6] Creating model comparison plots...")
    plot_model_comparison()

    # 2. Data distribution
    print("\n[2/6] Creating data distribution plots...")
    plot_data_distribution()

    # 3-5. Per-model visualizations
    print("\n[3/6] Creating predicted vs actual plots for all models...")
    for model_key in MODELS.keys():
        print(f"  Processing {MODELS[model_key]['name']}...")
        plot_predictions_vs_actual(model_key)

    print("\n[4/6] Creating residual analysis plots for all models...")
    for model_key in MODELS.keys():
        print(f"  Processing {MODELS[model_key]['name']}...")
        plot_residuals(model_key)

    print("\n[5/6] Creating ligand embedding visualizations for GNN models...")
    for model_key in MODELS.keys():
        if MODELS[model_key]['type'] == 'gnn':
            print(f"  Processing {MODELS[model_key]['name']}...")
            plot_ligand_embeddings(model_key)

    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: {VIZ_DIR}")
    print("\nGenerated visualizations:")
    print("  - Model comparisons: results/visualizations/comparisons/")
    print("  - Data analysis: results/visualizations/data_analysis/")
    print("  - Per-model plots: results/visualizations/{model_name}/")
    print("\nReady for your blog post!")


if __name__ == '__main__':
    main()
