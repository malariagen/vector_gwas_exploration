import pandas as pd
import numpy as np
import xarray as xr
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class AnalysisHelper:
    """A collection of helper functions for preparing data for GWAS models."""

    @staticmethod
    def _prepare_pca_data(variant_ds: xr.Dataset) -> np.ndarray:
        """Prepares the genotype matrix for PCA, handling missing values by imputation."""
        # Use the mean genotype across ploidy. Shape becomes (samples, variants)
        geno_matrix = variant_ds["call_genotype"].mean(dim="ploidy").values

        # Impute any remaining NaN values with the mean of the variant (column)
        if np.isnan(geno_matrix).any():
            col_means = np.nanmean(geno_matrix, axis=0)
            nan_inds = np.where(np.isnan(geno_matrix))
            geno_matrix[nan_inds] = np.take(col_means, nan_inds[1])

        return geno_matrix

    @staticmethod
    def compute_pca_components(
        variant_ds: xr.Dataset, n_components: int = 5
    ) -> pd.DataFrame:
        """Computes and returns a DataFrame of principal components."""

        geno_matrix = AnalysisHelper._prepare_pca_data(variant_ds)

        if geno_matrix.shape[1] < n_components:
            warnings.warn(
                f"Reducing PCA components to {geno_matrix.shape[1]} due to data limitations."
            )
            n_components = geno_matrix.shape[1]

        scaler = StandardScaler()
        geno_scaled = scaler.fit_transform(geno_matrix)

        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(geno_scaled)

        pc_cols = [f"PC{i+1}" for i in range(n_components)]
        pc_df = pd.DataFrame(pcs, columns=pc_cols, index=variant_ds.samples.values)

        return pc_df

    @staticmethod
    def prepare_modeling_dataframe(
        final_ds: xr.Dataset, add_pca: bool = True, n_components: int = 5
    ) -> pd.DataFrame:
        """
        Creates the final pandas DataFrame for statistical modeling, ensuring one row per sample.
        """
        df_dict = {
            "samples": final_ds.samples.values,
            "phenotype": final_ds.phenotype_binary.values,
        }

        # Create a pandas DataFrame from this dictionary.
        df = pd.DataFrame(df_dict)

        # 2. Add the binarized variant columns one by one.
        for var_name in final_ds.variants.values:
            # Check for non-wildtype allele across the ploidy dimension
            has_variant = (
                (final_ds["call_genotype"].sel(variants=var_name).values > 0)
                .any(axis=1)
                .astype(int)
            )
            # This correctly creates an array of length N (number of samples)
            df[f"has_{var_name}"] = has_variant

        # 3. Add PCA components if requested.
        if add_pca:
            pc_df = AnalysisHelper.compute_pca_components(final_ds, n_components)
            # Merge the PCs based on the sample ID
            df = df.merge(pc_df, left_on="samples", right_index=True)

        return df
