"""
Analyze train/val/test split distributions to identify potential data leakage
or distribution mismatch that could explain the 50% val→test performance gap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import pickle

def load_splits():
    """Load all three data splits"""
    train = pd.read_csv('data/splits/train.csv', low_memory=False)
    val = pd.read_csv('data/splits/val.csv', low_memory=False)
    test = pd.read_csv('data/splits/test.csv', low_memory=False)

    print(f"Train: {len(train)} samples")
    print(f"Val:   {len(val)} samples")
    print(f"Test:  {len(test)} samples")
    print()

    return train, val, test


def analyze_target_distribution(train, val, test):
    """Compare pKd distributions across splits"""
    print("="*70)
    print("1. TARGET DISTRIBUTION ANALYSIS (pKd)")
    print("="*70)

    # Basic statistics
    for name, df in [('Train', train), ('Val', val), ('Test', test)]:
        pkd = df['pKd'].dropna()
        print(f"\n{name}:")
        print(f"  Mean:   {pkd.mean():.3f}")
        print(f"  Median: {pkd.median():.3f}")
        print(f"  Std:    {pkd.std():.3f}")
        print(f"  Min:    {pkd.min():.3f}")
        print(f"  Max:    {pkd.max():.3f}")
        print(f"  Q1:     {pkd.quantile(0.25):.3f}")
        print(f"  Q3:     {pkd.quantile(0.75):.3f}")

    # Statistical tests
    print("\n" + "-"*70)
    print("Statistical Tests (Kolmogorov-Smirnov):")
    print("-"*70)

    train_pkd = train['pKd'].dropna()
    val_pkd = val['pKd'].dropna()
    test_pkd = test['pKd'].dropna()

    # Compare train vs val
    ks_stat_tv, p_value_tv = stats.ks_2samp(train_pkd, val_pkd)
    print(f"Train vs Val:  KS statistic = {ks_stat_tv:.4f}, p-value = {p_value_tv:.4e}")

    # Compare train vs test
    ks_stat_tt, p_value_tt = stats.ks_2samp(train_pkd, test_pkd)
    print(f"Train vs Test: KS statistic = {ks_stat_tt:.4f}, p-value = {p_value_tt:.4e}")

    # Compare val vs test (KEY!)
    ks_stat_vt, p_value_vt = stats.ks_2samp(val_pkd, test_pkd)
    print(f"Val vs Test:   KS statistic = {ks_stat_vt:.4f}, p-value = {p_value_vt:.4e}")

    if p_value_vt < 0.05:
        print("\n⚠️  WARNING: Val and Test have significantly different pKd distributions!")
        print("   This could explain the 50% performance gap.")
    else:
        print("\n✓ Val and Test have similar pKd distributions")

    print()


def analyze_protein_overlap(train, val, test):
    """Check for protein leakage between splits"""
    print("="*70)
    print("2. PROTEIN OVERLAP ANALYSIS")
    print("="*70)

    train_proteins = set(train['BindingDB Target Chain Sequence 1'].dropna())
    val_proteins = set(val['BindingDB Target Chain Sequence 1'].dropna())
    test_proteins = set(test['BindingDB Target Chain Sequence 1'].dropna())

    print(f"\nUnique proteins:")
    print(f"  Train: {len(train_proteins)}")
    print(f"  Val:   {len(val_proteins)}")
    print(f"  Test:  {len(test_proteins)}")

    # Check for leakage
    train_val_overlap = train_proteins & val_proteins
    train_test_overlap = train_proteins & test_proteins
    val_test_overlap = val_proteins & test_proteins

    print(f"\nOverlap (should be 0 for proper protein-level split):")
    print(f"  Train ∩ Val:  {len(train_val_overlap)} proteins")
    print(f"  Train ∩ Test: {len(train_test_overlap)} proteins")
    print(f"  Val ∩ Test:   {len(val_test_overlap)} proteins")

    if train_test_overlap:
        print(f"\n⚠️  CRITICAL: {len(train_test_overlap)} proteins appear in both train and test!")
        print("   This is DATA LEAKAGE and invalidates test results!")
    else:
        print("\n✓ No train-test protein leakage")

    if val_test_overlap:
        print(f"\n⚠️  WARNING: {len(val_test_overlap)} proteins appear in both val and test!")
    else:
        print("✓ No val-test protein leakage")

    print()


def analyze_ligand_overlap(train, val, test):
    """Check for ligand leakage between splits"""
    print("="*70)
    print("3. LIGAND OVERLAP ANALYSIS")
    print("="*70)

    train_ligands = set(train['Ligand SMILES'].dropna())
    val_ligands = set(val['Ligand SMILES'].dropna())
    test_ligands = set(test['Ligand SMILES'].dropna())

    print(f"\nUnique ligands:")
    print(f"  Train: {len(train_ligands)}")
    print(f"  Val:   {len(val_ligands)}")
    print(f"  Test:  {len(test_ligands)}")

    # Check for overlap (expected for ligands)
    train_val_overlap = train_ligands & val_ligands
    train_test_overlap = train_ligands & test_ligands
    val_test_overlap = val_ligands & test_ligands

    print(f"\nOverlap (expected for ligands - same drug, different protein):")
    print(f"  Train ∩ Val:  {len(train_val_overlap)} ligands ({100*len(train_val_overlap)/len(val_ligands):.1f}% of val)")
    print(f"  Train ∩ Test: {len(train_test_overlap)} ligands ({100*len(train_test_overlap)/len(test_ligands):.1f}% of test)")
    print(f"  Val ∩ Test:   {len(val_test_overlap)} ligands ({100*len(val_test_overlap)/len(test_ligands):.1f}% of test)")

    print()


def analyze_protein_ligand_pair_overlap(train, val, test):
    """Check for exact protein-ligand pair leakage"""
    print("="*70)
    print("4. PROTEIN-LIGAND PAIR OVERLAP ANALYSIS")
    print("="*70)

    def get_pairs(df):
        pairs = set()
        for _, row in df.iterrows():
            protein = row['BindingDB Target Chain Sequence 1']
            ligand = row['Ligand SMILES']
            if pd.notna(protein) and pd.notna(ligand):
                pairs.add((protein, ligand))
        return pairs

    train_pairs = get_pairs(train)
    val_pairs = get_pairs(val)
    test_pairs = get_pairs(test)

    print(f"\nUnique protein-ligand pairs:")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Val:   {len(val_pairs)}")
    print(f"  Test:  {len(test_pairs)}")

    # Check for leakage (critical issue!)
    train_val_overlap = train_pairs & val_pairs
    train_test_overlap = train_pairs & test_pairs
    val_test_overlap = val_pairs & test_pairs

    print(f"\nExact pair overlap (SHOULD BE 0!):")
    print(f"  Train ∩ Val:  {len(train_val_overlap)} pairs ({100*len(train_val_overlap)/len(val_pairs):.1f}% of val)")
    print(f"  Train ∩ Test: {len(train_test_overlap)} pairs ({100*len(train_test_overlap)/len(test_pairs):.1f}% of test)")
    print(f"  Val ∩ Test:   {len(val_test_overlap)} pairs ({100*len(val_test_overlap)/len(test_pairs):.1f}% of test)")

    if train_test_overlap:
        print(f"\n⚠️  CRITICAL: {len(train_test_overlap)} exact protein-ligand pairs in both train and test!")
        print("   This is severe data leakage!")
    else:
        print("\n✓ No exact pair leakage between train and test")

    print()


def analyze_protein_family_distribution(train, val, test):
    """Analyze if val/test have different protein families than train"""
    print("="*70)
    print("5. PROTEIN FAMILY DISTRIBUTION")
    print("="*70)

    # Use Target Name as proxy for protein family
    def get_target_names(df):
        return Counter(df['Target Name'].dropna())

    train_targets = get_target_names(train)
    val_targets = get_target_names(val)
    test_targets = get_target_names(test)

    print(f"\nUnique protein targets:")
    print(f"  Train: {len(train_targets)}")
    print(f"  Val:   {len(val_targets)}")
    print(f"  Test:  {len(test_targets)}")

    # Find top 10 targets in each split
    print("\nTop 10 targets in Train:")
    for target, count in train_targets.most_common(10):
        print(f"  {target[:50]:50s} {count:5d} samples")

    print("\nTop 10 targets in Test:")
    for target, count in test_targets.most_common(10):
        train_count = train_targets.get(target, 0)
        val_count = val_targets.get(target, 0)
        print(f"  {target[:50]:50s} {count:5d} test, {train_count:5d} train, {val_count:5d} val")

    # Check for completely novel targets in test
    test_only_targets = set(test_targets.keys()) - set(train_targets.keys())
    print(f"\nNovel targets in test (not in train): {len(test_only_targets)}")
    if test_only_targets:
        test_only_samples = sum(test_targets[t] for t in test_only_targets)
        print(f"  Affecting {test_only_samples} test samples ({100*test_only_samples/len(test):.1f}% of test set)")
        print("\n⚠️  Test set contains novel protein families not seen during training!")
        print("   This explains poor generalization.")

    print()


def analyze_sequence_length_distribution(train, val, test):
    """Compare protein sequence lengths across splits"""
    print("="*70)
    print("6. PROTEIN SEQUENCE LENGTH DISTRIBUTION")
    print("="*70)

    def get_lengths(df):
        return df['BindingDB Target Chain Sequence 1'].dropna().str.len()

    train_lengths = get_lengths(train)
    val_lengths = get_lengths(val)
    test_lengths = get_lengths(test)

    print("\nProtein sequence length statistics:")
    for name, lengths in [('Train', train_lengths), ('Val', val_lengths), ('Test', test_lengths)]:
        print(f"\n{name}:")
        print(f"  Mean:   {lengths.mean():.1f}")
        print(f"  Median: {lengths.median():.1f}")
        print(f"  Std:    {lengths.std():.1f}")
        print(f"  Min:    {lengths.min()}")
        print(f"  Max:    {lengths.max()}")

    # Statistical test
    ks_stat, p_value = stats.ks_2samp(val_lengths, test_lengths)
    print(f"\nKS test (Val vs Test): statistic = {ks_stat:.4f}, p-value = {p_value:.4e}")

    if p_value < 0.05:
        print("⚠️  Val and Test have different protein length distributions")
    else:
        print("✓ Val and Test have similar protein length distributions")

    print()


def analyze_ligand_size_distribution(train, val, test):
    """Compare ligand sizes (molecular weight) across splits"""
    print("="*70)
    print("7. LIGAND SIZE DISTRIBUTION")
    print("="*70)

    # Use number of atoms as proxy for molecular size
    def count_atoms(smiles):
        """Count non-hydrogen atoms in SMILES"""
        if pd.isna(smiles):
            return None
        # Simple heuristic: count uppercase letters (atoms)
        return sum(1 for c in smiles if c.isupper())

    train['num_atoms'] = train['Ligand SMILES'].apply(count_atoms)
    val['num_atoms'] = val['Ligand SMILES'].apply(count_atoms)
    test['num_atoms'] = test['Ligand SMILES'].apply(count_atoms)

    print("\nLigand size (atom count) statistics:")
    for name, df in [('Train', train), ('Val', val), ('Test', test)]:
        atoms = df['num_atoms'].dropna()
        print(f"\n{name}:")
        print(f"  Mean:   {atoms.mean():.1f}")
        print(f"  Median: {atoms.median():.1f}")
        print(f"  Std:    {atoms.std():.1f}")
        print(f"  Min:    {atoms.min():.0f}")
        print(f"  Max:    {atoms.max():.0f}")

    # Statistical test
    ks_stat, p_value = stats.ks_2samp(val['num_atoms'].dropna(), test['num_atoms'].dropna())
    print(f"\nKS test (Val vs Test): statistic = {ks_stat:.4f}, p-value = {p_value:.4e}")

    if p_value < 0.05:
        print("⚠️  Val and Test have different ligand size distributions")
    else:
        print("✓ Val and Test have similar ligand size distributions")

    print()


def analyze_binding_affinity_by_protein_family(train, val, test):
    """Check if val/test protein families have different binding characteristics"""
    print("="*70)
    print("8. BINDING AFFINITY BY PROTEIN FAMILY")
    print("="*70)

    def get_family_affinities(df):
        """Get mean pKd for each protein family"""
        family_stats = {}
        grouped = df.groupby('Target Name')
        for family, group in grouped:
            if len(group) >= 5:  # Only families with 5+ samples
                family_stats[family] = {
                    'mean_pkd': group['pKd'].mean(),
                    'std_pkd': group['pKd'].std(),
                    'count': len(group)
                }
        return family_stats

    train_families = get_family_affinities(train)
    test_families = get_family_affinities(test)

    # Find families that appear in test but have very different binding profiles
    print("\nProtein families with different binding profiles in test vs train:")
    print("(Families with |mean_pkd_test - mean_pkd_train| > 1.0)")
    print()

    mismatches = []
    for family, test_stats in test_families.items():
        if family in train_families:
            train_mean = train_families[family]['mean_pkd']
            test_mean = test_stats['mean_pkd']
            diff = abs(test_mean - train_mean)

            if diff > 1.0:  # Significant difference
                mismatches.append({
                    'family': family,
                    'train_mean': train_mean,
                    'test_mean': test_mean,
                    'diff': diff,
                    'test_count': test_stats['count']
                })

    if mismatches:
        mismatches.sort(key=lambda x: x['test_count'], reverse=True)
        for m in mismatches[:10]:  # Top 10
            print(f"  {m['family'][:50]:50s}")
            print(f"    Train mean pKd: {m['train_mean']:.2f}")
            print(f"    Test mean pKd:  {m['test_mean']:.2f}")
            print(f"    Difference:     {m['diff']:.2f}")
            print(f"    Test samples:   {m['test_count']}")
            print()

        total_affected = sum(m['test_count'] for m in mismatches)
        print(f"Total test samples affected: {total_affected} ({100*total_affected/len(test):.1f}% of test set)")
        print("\n⚠️  Test set protein families have different binding characteristics!")
    else:
        print("✓ No major differences in binding profiles between train and test families")

    print()


def main():
    print("\n" + "="*70)
    print("DATA SPLIT DISTRIBUTION ANALYSIS")
    print("Investigating the 50% Val→Test performance gap")
    print("="*70)
    print()

    # Load data
    train, val, test = load_splits()

    # Run all analyses
    analyze_target_distribution(train, val, test)
    analyze_protein_overlap(train, val, test)
    analyze_ligand_overlap(train, val, test)
    analyze_protein_ligand_pair_overlap(train, val, test)
    analyze_protein_family_distribution(train, val, test)
    analyze_sequence_length_distribution(train, val, test)
    analyze_ligand_size_distribution(train, val, test)
    analyze_binding_affinity_by_protein_family(train, val, test)

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey things to look for:")
    print("  1. ⚠️  Data leakage (proteins/pairs appearing in train AND test)")
    print("  2. ⚠️  Different pKd distributions (KS test p-value < 0.05)")
    print("  3. ⚠️  Novel protein families in test set")
    print("  4. ⚠️  Different binding characteristics by family")
    print()


if __name__ == '__main__':
    main()
