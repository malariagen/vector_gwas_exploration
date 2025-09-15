import pandas as pd
import numpy as np
import os

def generate_all_mock_data():
    """
    Generates a two-part mock dataset for the GWAS Explorer:
    1. A full genome-wide scan result.
    2. A smaller file with detailed verification results for significant hits.
    """
    print("--- Starting Realistic Mock Data Generation ---")

    output_dir = '../data/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Generate Genome-Wide Scan Data---
    print("Simulating genome-wide scan data...")
    contigs = ['2L', '2R', '3L', '3R', 'X']
    contig_lengths = {'2L': 49_364_325, '2R': 61_545_105, '3L': 41_963_435, '3R': 53_200_684, 'X': 24_393_108}
    n_snps_per_chrom = 100_000
    all_dfs = []
    for contig in contigs:
        pos = np.sort(np.random.randint(1, contig_lengths[contig], n_snps_per_chrom))
        p_values = 10**(-np.random.uniform(0, 2.5, n_snps_per_chrom))
        df = pd.DataFrame({'contig': contig, 'pos': pos, 'p_value': p_values})
        all_dfs.append(df)
    gwas_full_df = pd.concat(all_dfs, ignore_index=True)
    gwas_full_df.loc[(gwas_full_df['contig'] == '2L') & (gwas_full_df['pos'] > 2_300_000) & (gwas_full_df['pos'] < 2_500_000), 'p_value'] = np.random.uniform(1e-9, 1e-6, gwas_full_df.loc[(gwas_full_df['contig'] == '2L') & (gwas_full_df['pos'] > 2_300_000) & (gwas_full_df['pos'] < 2_500_000)].shape[0])
    gwas_full_df.loc[(gwas_full_df['contig'] == '3R') & (gwas_full_df['pos'] > 21_300_000) & (gwas_full_df['pos'] < 21_500_000), 'p_value'] = np.random.uniform(1e-8, 1e-5, gwas_full_df.loc[(gwas_full_df['contig'] == '3R') & (gwas_full_df['pos'] > 21_300_000) & (gwas_full_df['pos'] < 21_500_000)].shape[0])
    gwas_full_df['-log10(p)'] = -np.log10(gwas_full_df['p_value'])
    gwas_full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    gwas_full_df.dropna(subset=['-log10(p)'], inplace=True)
    output_path_full = os.path.join(output_dir, 'mock_gwas_full_scan.csv')
    gwas_full_df.to_csv(output_path_full, index=False)
    print(f"Generated {len(gwas_full_df):,} genome-wide SNP results -> {output_path_full}")

    # --- 2. Generate Verification Hits Data ---
    print("Simulating verification data for top hits...")
    # Select only the most significant SNPs from our "hit" regions
    significant_hits_df = gwas_full_df[gwas_full_df['-log10(p)'] > 5].copy()
    n_hits = len(significant_hits_df)

    # Simulate results for ONLY the Mixed-Effects model
    significant_hits_df['log_odds_mixed'] = np.random.normal(1.2, 0.3, n_hits)
    significant_hits_df['ci_lower_mixed'] = significant_hits_df['log_odds_mixed'] - np.random.uniform(0.1, 0.3, n_hits)
    significant_hits_df['ci_upper_mixed'] = significant_hits_df['log_odds_mixed'] + np.random.uniform(0.1, 0.3, n_hits)
    
    output_path_verified = os.path.join(output_dir, 'mock_verification_hits.csv')
    significant_hits_df.to_csv(output_path_verified, index=False)
    print(f"Generated verification data for {n_hits} significant SNPs -> {output_path_verified}")

    # --- 3. Generate Gene Annotation Data ---
    print("Creating mock gene annotation data...")
    gene_data = {'contig': ['2L', '2L', '3R'], 'start': [2_358_158, 2_800_000, 21_400_000], 'end': [2_431_617, 2_805_000, 21_408_000], 'gene_name': ['Vgsc', 'GeneB', 'GeneC'], 'strand': ['+', '-', '+']}
    gene_annotations_df = pd.DataFrame(gene_data)
    output_path_genes = os.path.join(output_dir, 'mock_gene_annotations.csv')
    gene_annotations_df.to_csv(output_path_genes, index=False)
    print(f"Saved mock gene annotations -> {output_path_genes}")
    print("\n--- Mock Data Generation Complete ---")

if __name__ == '__main__':
    generate_all_mock_data()