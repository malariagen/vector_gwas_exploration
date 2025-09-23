import pandas as pd
import xarray as xr
import warnings
from typing import Optional


class PhenotypeHelper:
    @staticmethod
    def create_binary_series(df: pd.DataFrame) -> pd.Series:
        """
        Convert a DataFrame with a 'phenotype' column into a binary series.
        Maps known phenotype strings to 0/1 and retains sample_id as index.
        """
        if "phenotype" not in df.columns:
            raise ValueError("Missing 'phenotype' column in DataFrame")

        phenotype_map = {
            "alive": 1,
            "dead": 0,
            "survived": 1,
            "died": 0,
            "resistant": 1,
            "susceptible": 0,
        }
        phenotype_lower = df["phenotype"].astype(str).str.lower()
        binary = phenotype_lower.map(phenotype_map)

        if binary.isna().any():
            unmapped = df.loc[binary.isna(), "phenotype"].unique()
            warnings.warn(f"Unmapped phenotype values: {list(unmapped)}")

        # Use sample_id as index if available
        if "sample_id" in df.columns:
            binary.index = df["sample_id"].values
        return binary.astype(float)

    @staticmethod
    def create_dataset(
        df: pd.DataFrame, variant_data: Optional[xr.Dataset] = None
    ) -> xr.Dataset:
        """
        Build an xarray.Dataset combining phenotype data and optional variant dataset.
        """
        binary = PhenotypeHelper.create_binary_series(df)
        df_idx = df.set_index("sample_id")
        sample_ids = df_idx.index.values

        data_vars = {
            "phenotype_binary": ("samples", binary.reindex(sample_ids).values),
            "phenotype": ("samples", df_idx["phenotype"].values),
            "insecticide": (
                "samples",
                df_idx.get(
                    "insecticide", pd.Series(["simulated"] * len(df_idx))
                ).values,
            ),
            "dose": (
                "samples",
                df_idx.get("dose", pd.Series(["simulated"] * len(df_idx))).values,
            ),
        }
        # Optional metadata columns
        for col in ["location", "country", "collection_date", "species", "sample_set"]:
            if col in df_idx.columns:
                data_vars[col] = ("samples", df_idx[col].values)

        ds = xr.Dataset(data_vars, coords={"samples": sample_ids})

        # Merge with variant_data if provided
        if variant_data is not None:
            try:
                coord = next(
                    c for c in ["samples", "sample_id"] if c in variant_data.coords
                )
                common = list(set(sample_ids) & set(variant_data.coords[coord].values))
                ds = ds.sel(samples=common)
                var_ds = variant_data.sel({coord: common}).rename({coord: "samples"})
                ds = xr.merge([ds, var_ds])
            except Exception as e:
                warnings.warn(f"Failed to merge variant data: {e}")
        return ds
