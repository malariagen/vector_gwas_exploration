import pandas as pd
import numpy as np
import xarray as xr
import scipy.stats
import warnings

from malariagen_data import Ag3
from utils.gwas_utils import parse_regions


class GWASScanner:
    """
    Performs a genome-wide scan using a fast statistical test (Chi-squared)
    to identify candidate SNPs for further analysis.
    """

    def __init__(self, ag3: Ag3):
        """
        Initializes the scanner with a malariagen_data Ag3 object.
        """
        self.ag3 = ag3
        self.results_df = None

    def _prepare_phenotypes(self, insecticide: str) -> pd.Series:
        """
        Loads and prepares a clean pandas Series of binary phenotypes.
        """
        print(f"Loading and preparing phenotypes for {insecticide}...")
        phenotype_sample_sets = self.ag3.phenotype_sample_sets()
        pheno_series = self.ag3.phenotype_binary(
            sample_sets=phenotype_sample_sets,
            insecticide=insecticide
        ).dropna().astype(int)

        if pheno_series.empty:
            raise ValueError(f"No valid phenotype data found for insecticide '{insecticide}'.")
        
        print(f"Found {len(pheno_series)} samples with valid phenotype data.")
        return pheno_series

    def run_scan(self, 
                 insecticide: str, 
                 region: str | list[str] = None,
                 chunk_size: int = 1_000_000):
        """
        Runs a genome-wide or regional scan using a Chi-squared test.
        """
        pheno_series = self._prepare_phenotypes(insecticide)
        pheno_sample_ids = pheno_series.index.tolist()
        sample_query = f"sample_id in {pheno_sample_ids}"
        
        results = []
        regions_to_scan = parse_regions(self.ag3, region)

        for contig, start_pos, end_pos in regions_to_scan:
            print(f"\n--- Processing region {contig}:{start_pos}-{end_pos} ---")
            
            try:
                for start in range(start_pos, end_pos + 1, chunk_size):
                    chunk_end = min(start + chunk_size - 1, end_pos)
                    chunk_region = f"{contig}:{start}-{chunk_end}"
                    
                    print(f"Scanning chunk: {chunk_region}...")

                    ds_chunk = self.ag3.snp_calls(
                        region=chunk_region, 
                        sample_query=sample_query
                    ).compute()

                    if ds_chunk.sizes['variants'] == 0:
                        continue
                    
                    
                    # 1. Binarize genotype data. Shape remains (n_variants, n_samples). NO TRANSPOSE.
                    has_variant_matrix = (ds_chunk['call_genotype'].values > 0).any(axis=2)
                    
                    # 2. Align phenotypes to the sample order of the genotype chunk.
                    phenotypes_aligned = pheno_series.loc[ds_chunk.sample_id.values].values

                    # 3. Get the positions for this chunk.
                    positions = ds_chunk.variant_position.values

                    # 4. Iterate through SNPs (now the first axis of the matrix).
                    for i in range(has_variant_matrix.shape[0]): # Loop over variants
                        geno_vector = has_variant_matrix[i, :] # Get the i-th SNP for all samples
                        
                        # Create the 2x2 contingency table manually
                        table = np.zeros((2, 2), dtype=int)
                        for p, g in zip(phenotypes_aligned, geno_vector):
                            table[p, int(g)] += 1
                        
                        if np.all(table.sum(axis=0) > 0) and np.all(table.sum(axis=1) > 0):
                            chi2, p, dof, expected = scipy.stats.chi2_contingency(table, correction=False)
                            results.append({'contig': contig, 'pos': positions[i], 'p_value': p})
                    

            except Exception as e:
                print(f"  Could not process region {contig}:{start_pos}-{end_pos} due to error: {e}")

        self.results_df = pd.DataFrame(results)
        if not self.results_df.empty:
            self.results_df['p_value'] = self.results_df['p_value'].replace(0, np.finfo(float).tiny)
            self.results_df['-log10(p)'] = -np.log10(self.results_df['p_value'])
        
        print("\n--- Scan complete. ---")
        return self.results_df