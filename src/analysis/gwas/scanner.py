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

        Parameters
        ----------
        ag3 : Ag3
            An instantiated Ag3 data resource object.
        """
        self.ag3 = ag3
        self.results_df = None

    def _prepare_phenotypes(self, insecticide: str, sample_sets: list) -> pd.Series:
        """
        Loads and prepares a clean pandas Series of binary phenotypes.
        """
        print(f"Loading and preparing phenotypes for {insecticide}...")
        pheno_series = self.ag3.phenotype_binary(
            sample_sets=sample_sets,
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

        Parameters
        ----------
        insecticide : str
            The insecticide to use for phenotype filtering (e.g., 'Deltamethrin').
        region : str or list of str, optional
            The genomic region(s) to scan. Can be a contig name ('2L'), a region
            string ('2L:1-10,000,000'), or a list of these. 
            If None, scans all major chromosome arms.
        chunk_size : int, optional
            The size of the genomic window to process in each iteration.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the scan results (contig, pos, p_value).
        """
        phenotype_sample_sets = self.ag3.phenotype_sample_sets()
        pheno_series = self._prepare_phenotypes(insecticide, phenotype_sample_sets)
        
        pheno_df = pheno_series.to_frame(name='phenotype')
        pheno_sample_ids = pheno_df.index.tolist()
        sample_query = f"sample_id in {pheno_sample_ids}"
        
        results = []
        
        # Use the new helper function to parse the user's region input
        regions_to_scan = parse_regions(self.ag3, region)

        # The main loop now iterates over a standardized list of (contig, start, end) tuples
        for contig, start_pos, end_pos in regions_to_scan:
            print(f"\n--- Processing region {contig}:{start_pos}-{end_pos} ---")
            
            try:
                # Iterate through the region in chunks
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
                    
                    # Binarize the genotype data
                    has_variant_matrix = (ds_chunk['call_genotype'].values > 0).any(axis=2)
                    
                    # Manually create the genotype DataFrame, transposing the data
                    geno_df = pd.DataFrame(
                        data=has_variant_matrix.T,
                        index=ds_chunk.sample_id.values,
                        columns=ds_chunk.variant_position.values
                    )
                    
                    # Join with phenotype data
                    combined_df = pheno_df.join(geno_df, how='inner')

                    # Iterate through the SNPs in the chunk
                    for pos_col in geno_df.columns:
                        contingency_table = pd.crosstab(
                            combined_df['phenotype'], 
                            combined_df[pos_col]
                        )
                        
                        if contingency_table.shape == (2, 2):
                            chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table, correction=False)
                            results.append({'contig': contig, 'pos': pos_col, 'p_value': p})

            except Exception as e:
                print(f"  Could not process region {contig}:{start_pos}-{end_pos} due to error: {e}")

        # Finalize and return the results DataFrame
        self.results_df = pd.DataFrame(results)
        if not self.results_df.empty:
            self.results_df['p_value'] = self.results_df['p_value'].replace(0, np.finfo(float).tiny)
            self.results_df['-log10(p)'] = -np.log10(self.results_df['p_value'])
        
        print("\n--- Scan complete. ---")
        return self.results_df