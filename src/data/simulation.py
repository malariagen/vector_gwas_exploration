import pandas as pd
import numpy as np
import xarray as xr

from malariagen_data import Ag3
from src.utils.phenotype_utils import PhenotypeHelper


class ResistanceSimulator:
    """
    Simulator for insecticide resistance phenotypes based on genetic variants.
    """
    def __init__(self):
        self.ag3 = Ag3()
        self.meta = self.ag3.sample_metadata()
        self.sample_ids = self.meta["sample_id"].tolist()
        self.n = len(self.sample_ids)

    def fetch_variant(self, region: str, pos: int, fallback: bool = True):
        """
        Fetch genotype calls for a SNP around `pos` in `region`.
        Returns (genotypes_array, variant_id).
        """
        try:
            ds = self.ag3.snp_calls(region=region, sample_sets="ag3.0").sel(samples=self.sample_ids)
            idx = np.argmin(np.abs(ds["variant_position"].values - pos))
            genos = ds["call_genotype"].isel(variants=idx).values
            vid = ds["variants"].isel(variants=idx).item()
            return genos, vid
        except Exception:
            if fallback:
                return np.random.randint(0, 3, self.n), "random_variant"
            raise

    def combine_snp_data(self, geno1: np.ndarray, geno2: np.ndarray) -> xr.Dataset:
        """
        Combine two genotype arrays into a single xarray Dataset with two variants.
        """
        data = np.stack([geno1, geno2], axis=1)
        return xr.Dataset(
            {"call_genotype": (("samples", "variants"), data)},
            coords={
                "samples": self.sample_ids,
                "variants": ["VGSC_L995F_proxy", "Ace1_RDL_proxy"],
                "variant_position": ("variants", [2438000, 3484107]),
                "variant_contig": ("variants", ["2L", "2R"]),
            }
        )

    def simulate_phenotypes(
        self,
        geno1: np.ndarray,
        geno2: np.ndarray,
        base_prob: float = 0.15,
        effect1: float = 0.6,
        effect2: float = 0.2,
        country_effects: dict = None
    ) -> pd.Series:
        """
        Simulate binary resistance phenotypes given genotype arrays and effect sizes.
        """
        if country_effects is None:
            country_effects = {
                'Ghana': 0.10, 'Burkina Faso': 0.05, 'Uganda': -0.05,
                'Mali': -0.10, 'Nigeria': 0.08, 'Kenya': 0.03
            }
        has1 = (geno1 >= 1).astype(int)
        has2 = (geno2 >= 1).astype(int)
        probs = []
        for i, sid in enumerate(self.sample_ids):
            country = self.meta.loc[self.meta['sample_id'] == sid, 'country'].iloc[0]
            p = base_prob + has1[i]*effect1 + has2[i]*effect2 + country_effects.get(country, 0)
            probs.append(np.clip(p, 0.01, 0.99))
        return pd.Series((np.random.rand(self.n) < probs).astype(int), index=self.sample_ids)

    def create_phenotype_df(self, phenos: pd.Series) -> pd.DataFrame:
        """
        Build a DataFrame from simulated phenotypes for downstream dataset creation.
        """
        return pd.DataFrame({
            "sample_id": phenos.index,
            "phenotype": phenos.map({0: "susceptible", 1: "resistant"}),
            "insecticide": "simulated_insecticide",
            "dose": "simulated_dose"
        })

