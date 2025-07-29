import pandas as pd
import numpy as np
import xarray as xr
import scipy.stats
import warnings
from malariagen_data import Ag3
from utils.analysis_utils import AnalysisHelper

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
        Runs a genome-wide or regional scan.
        """
        phenotype_sample_sets = self.ag3.phenotype_sample_sets()
        pheno_series = self._prepare_phenotypes(insecticide, phenotype_sample_sets)
        
        pheno_df = pheno_series.to_frame(name='phenotype')
        pheno_sample_ids = pheno_df.index.tolist()
        
        sample_query = f"sample_id in {pheno_sample_ids}"
        results = []
        
        if region is None:
            regions_to_scan = self.ag3.contigs
        elif isinstance(region, str):
            regions_to_scan = [region]
        else:
            regions_to_scan = region

        for r in regions_to_scan:
            print(f"\n--- Processing region {r} ---")
            
            try:
                # Determine bounds for chunking
                if ":" in r:
                    contig = r.split(":")[0]
                    start_pos, end_pos = map(int, r.split(":")[1].replace(",", "").split("-"))
                else:
                    contig = r
                    start_pos = 1
                    end_pos = self.ag3.genome_sequence(contig).shape[0]

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
                    
                    # 1. Binarize the genotype data into a numpy array
                    has_variant_matrix = (ds_chunk['call_genotype'].values > 0).any(axis=2)
                    
                    # 2. Manually create the genotype DataFrame
                    geno_df = pd.DataFrame(
                        data= has_variant_matrix.T,
                        index=ds_chunk.sample_id.values,
                        columns=ds_chunk.variant_position.values
                    )
                    
                    # 3. Join the phenotype and genotype dataframes.
                    combined_df = pheno_df.join(geno_df, how='inner')

                    # 4. Iterate through the SNP columns of the combined dataframe
                    for pos_col in geno_df.columns:
                        
                        contingency_table = pd.crosstab(
                            combined_df['phenotype'], 
                            combined_df[pos_col]
                        )
                        
                        if contingency_table.shape == (2, 2):
                            chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table, correction=False)
                            results.append({'contig': contig, 'pos': pos_col, 'p_value': p})

            except Exception as e:
                print(f"  Could not process region {r} due to error: {e}")

        # Finalize and return the results DataFrame
        self.results_df = pd.DataFrame(results)
        if not self.results_df.empty:
            self.results_df['p_value'] = self.results_df['p_value'].replace(0, np.finfo(float).tiny)
            self.results_df['-log10(p)'] = -np.log10(self.results_df['p_value'])
        
        print("\n--- Scan complete. ---")
        return self.results_df

class GWASVerifier:
    """
    Takes top hits from a GWAS scan and verifies them using a specified statistical model.
    This class is model-agnostic and expects a model object with .fit() and .get_params() methods.
    """

    def __init__(self, ag3: Ag3, analysis_helper: AnalysisHelper):
        """
        Initializes the verifier.

        Parameters
        ----------
        ag3 : Ag3
            An instantiated Ag3 data resource object.
        analysis_helper : AnalysisHelper
            An instance of your AnalysisHelper class.
        """
        self.ag3 = ag3
        self.helper = analysis_helper
        self.results_df = None

    def _get_top_hits(self, scan_results_df: pd.DataFrame, n_hits: int = 1000) -> pd.DataFrame:
        """Selects the top N hits from the scan results."""
        top_hits = scan_results_df.nsmallest(n_hits, 'p_value').reset_index(drop=True)
        print(f"Selected top {len(top_hits)} candidate SNPs for verification.")
        return top_hits

    def verify_hits(self, 
                    scan_results_df: pd.DataFrame, 
                    pheno_series: pd.Series, 
                    model: object,
                    n_hits: int = 1000, 
                    n_pca_components: int = 5):
        """
        Runs the verification pipeline on the top hits using the provided model.

        Parameters
        ----------
        scan_results_df : pd.DataFrame
            The DataFrame of p-values from the GWASScanner.
        pheno_series : pd.Series
            A pandas Series of binary phenotypes, indexed by sample_id.
        model : object
            An instantiated model object which must have a .fit() method that accepts
            (analysis_df, variant_names, pc_names) and a .get_params() method.
        n_hits : int, optional
            The number of top hits to verify.
        n_pca_components : int, optional
            The number of principal components to use for correction.

        Returns
        -------
        pd.DataFrame
            A DataFrame with detailed model results for the top hits.
        """
        top_hits_df = self._get_top_hits(scan_results_df, n_hits)
        pheno_sample_ids = pheno_series.index.tolist()
        sample_query = f"sample_id in {pheno_sample_ids}"

        # Step 1: Load Genotype Data for ALL Top Hits at once
        regions = [f"{row.contig}:{row.pos}-{row.pos}" for _, row in top_hits_df.iterrows()]
        print("Loading genotype data for all top hits...")
        ds_top_hits = self.ag3.snp_calls(
            region=regions,
            sample_query=sample_query
        ).compute()

        # Step 2: Compute PCA ONCE on the top hits
        print("Computing PCA on top hits for population structure correction...")
        pc_df = self.helper.compute_pca_components(ds_top_hits, n_components=n_pca_components)
        pc_names = pc_df.columns.tolist()

        # Step 3: Iterate and verify each hit with the provided model
        print(f"Verifying each hit with the {type(model).__name__} model...")
        verification_results = []
        
        for _, hit in top_hits_df.iterrows():
            pos = hit['pos']
            contig = hit['contig']
            
            # Find the variant in our dataset that corresponds to this hit
            variant_mask = (ds_top_hits.variant_position.values == pos) & (ds_top_hits.variant_contig.values == contig)
            
            if not np.any(variant_mask):
                continue

            variant_data = ds_top_hits.isel(variants=variant_mask)
            
            # The variant name in the dataset is its position, let's use that
            variant_name = f"{contig}_{pos}"
            variant_data = variant_data.assign_coords(variants=[variant_name])

            # Use your validated helper to create the modeling frame for this one SNP
            df_model = self.helper.prepare_modeling_dataframe(
                final_ds=variant_data,
                add_pca=False # We will add PCs manually
            )
            df_model = df_model.merge(pc_df, on='samples')

            if df_model[f'has_{variant_name}'].var() == 0:
                continue

            # Fit the provided model
            model.fit(df_model, variant_names=[variant_name], pc_names=pc_names)
            params = model.get_params().loc[f'has_{variant_name}']
            
            params['contig'] = contig
            params['pos'] = pos
            params['p_value_adj'] = params['p_value'] # Placeholder for adjusted p-value
            params['-log10(p_adj)'] = -np.log10(params['p_value'])
            verification_results.append(params)

        self.results_df = pd.DataFrame(verification_results)
        print("\n--- Verification complete. ---")
        return self.results_df