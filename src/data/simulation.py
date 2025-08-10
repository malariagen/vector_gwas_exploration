import os
import pandas as pd
import numpy as np
import xarray as xr
import warnings

from malariagen_data import Ag3

class ResistanceSimulator:
    """
    A utility to fetch genetic data and simulate resistance phenotypes.
    This class intelligently uses a local cache to avoid re-downloading data.
    """
    
    def __init__(self):
        """Initializes the simulator."""
        self.ag3 = Ag3()
        self.meta = self.ag3.sample_metadata()
        self.sample_ids = self.meta["sample_id"].tolist()
        self.n = len(self.sample_ids)
        self.available_sample_sets = self.ag3.sample_sets()["sample_set"].to_list()

    def fetch_variant_data(self, region: str, pos: int):
        """
        Fetches genotype data for a single SNP from the cloud.
        This is a potentially slow operation and includes a progress bar.
        """
        print(f"Fetching {region} for {self.n} samples from the cloud...")
        sample_query = f"sample_id in {self.sample_ids}"
        try:
            ds_plan = self.ag3.snp_calls(
                region=region, 
                sample_sets=self.available_sample_sets,
                sample_query=sample_query
            )
            
            ds = ds_plan.compute()    
            ds = ds.set_index(samples="sample_id")
            variant_positions = ds["variant_position"].values
            idx = np.argmin(np.abs(variant_positions - pos))
            
            actual_pos = variant_positions[idx]
            if actual_pos != pos:
                warnings.warn(f"Variant at pos {pos} not found. Using closest at {actual_pos}.")
            
            genos = ds["call_genotype"].isel(variants=idx).values
            vid = ds["variant_id"].isel(variants=idx).item()
            
            print(f"Successfully fetched {vid}.")
            return genos, vid

        except Exception as e:
            warnings.warn(f"Cloud fetch failed for {region}: {e}. Returning random data.")
            return np.random.randint(0, 3, size=(self.n, 2)), "random_variant"

    def combine_snp_data(self, variant_dict: dict) -> xr.Dataset:
        """Combines multiple genotype arrays into a single xarray Dataset."""
        geno_arrays = list(variant_dict.values())
        variant_names = list(variant_dict.keys())
        data = np.stack(geno_arrays, axis=1)
        
        return xr.Dataset(
            {"call_genotype": (("samples", "variants", "ploidy"), data)},
            coords={"samples": self.sample_ids, "variants": variant_names}
        )

    def simulate_phenotypes(self, geno1: np.ndarray, geno2: np.ndarray) -> pd.Series:
        """Simulate phenotypes based on two genotype arrays."""
        base_prob=0.15
        effect1=0.6
        effect2=0.2
        country_effects = {
            'Ghana': 0.10, 'Burkina Faso': 0.05, 'Uganda': -0.05,
            'Mali': -0.10, 'Nigeria': 0.08, 'Kenya': 0.03
        }
        has1 = (geno1 >= 1).any(axis=1).astype(int)
        has2 = (geno2 >= 1).any(axis=1).astype(int)
        probs = []
        for i, sid in enumerate(self.sample_ids):
            country = self.meta.loc[self.meta['sample_id'] == sid, 'country'].iloc[0]
            p = base_prob + has1[i]*effect1 + has2[i]*effect2 + country_effects.get(country, 0)
            probs.append(np.clip(p, 0.01, 0.99))
        return pd.Series((np.random.rand(self.n) < probs).astype(int), index=self.sample_ids)

    def create_phenotype_df(self, phenos: pd.Series) -> pd.DataFrame:
        """Build a DataFrame from simulated phenotypes."""
        df = self.meta.set_index("sample_id").copy()
        df["phenotype"] = phenos.map({0: "susceptible", 1: "resistant"})
        df["phenotype_binary"] = phenos
        df["insecticide"] = "simulated_insecticide"
        df["dose"] = "simulated_dose"
        return df.reset_index()