import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RESULTS_DIR = Path('results/stratified_split')
VIZ_DIR = Path('results/visualizations')

MODELS = {
    'xgboost': {
        'name': 'XGBoost',
        'results_file': 'xgboost_results.json',
        'color': '#e74c3c',
    },
    'gnn': {
        'name': 'GNN',
        'results_file': 'gnn_results.json',
        'color': '#3498db',
    },
    'gann': {
        'name': 'GANN (Attention)',
        'results_file': 'gann_results.json',
        'color': '#9b59b6',
    },
    'gnn_iter3': {
        'name': 'GNN Iter3',
        'results_file': 'iter3_results.json',
        'color': '#2ecc71',
    },
    'gnn_iter4': {
        'name': 'GNN Iter4',
        'results_file': 'iter4_results.json',
        'color': '#f39c12',
    },
    'gnn_iter5': {
        'name': 'GNN Iter5',
        'results_file': 'iter5_results.json',
        'color': '#1abc9c',
    }
}


def load_results(model_key):
    results_path = RESULTS_DIR / MODELS[model_key]['results_file']
    if not results_path.exists():
        return None

    with open(results_path, 'r') as f:
        return json.load(f)


def plot_model_comparison():
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
        return

    df = pd.DataFrame(models_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['name'], df['rmse'], color=df['color'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison - RMSE (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'comparisons/rmse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['name'], df['r2'], color=df['color'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison - R² (Higher is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(df['r2']) * 1.2])
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'comparisons/r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

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


def plot_data_distribution():
    train_df = pd.read_csv('data/splits/train.csv', low_memory=False)
    val_df = pd.read_csv('data/splits/val.csv', low_memory=False)
    test_df = pd.read_csv('data/splits/test.csv', low_memory=False)

    train_df = train_df[np.isfinite(train_df['pKd'])]
    val_df = val_df[np.isfinite(val_df['pKd'])]
    test_df = test_df[np.isfinite(test_df['pKd'])]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(train_df['pKd'], bins=50, alpha=0.6, label='Train', color='#3498db', edgecolor='black')
    ax.hist(val_df['pKd'], bins=50, alpha=0.6, label='Validation', color='#e74c3c', edgecolor='black')
    ax.hist(test_df['pKd'], bins=50, alpha=0.6, label='Test', color='#2ecc71', edgecolor='black')
    ax.set_xlabel('pKd (Binding Affinity)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('pKd Distribution Across Splits', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

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


def main():
    plot_model_comparison()
    plot_data_distribution()


if __name__ == '__main__':
    main()
