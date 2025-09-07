import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

class BayesianModel:
    """
    A Bayesian logistic regression model for GWAS verification using PyMC.

    This model estimates the probability of resistance using a logistic function,
    incorporating effects from genetic variants and population structure (PCs).
    It includes logic for using informative priors on known resistance genes to
    leverage existing biological knowledge.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initializes the Bayesian model.

        Parameters
        ----------
        random_seed : int, optional
            A seed for the random number generator for reproducibility of MCMC sampling.
        """
        self.random_seed = random_seed
        self.model = None
        self.idata = None  # Stores InferenceData from ArviZ
        self.summary_df = None

    def fit(self, analysis_df: pd.DataFrame, variant_names: list, pc_names: list):
        """
        Fits the Bayesian model to the provided data using MCMC sampling.

        Parameters
        ----------
        analysis_df : pd.DataFrame
            A DataFrame containing the phenotype, binary-encoded variant data
            (e.g., 'has_variant_X'), and principal components.
            The phenotype column must be named 'phenotype'.
        variant_names : list
            A list of column names corresponding to the genetic variants.
        pc_names : list
            A list of column names corresponding to the principal components.
        """
        phenotype = analysis_df['phenotype'].values
        X_variants = analysis_df[variant_names].values
        X_pcs = analysis_df[pc_names].values

        coords = {
            "variant_predictors": variant_names,
            "pc_predictors": pc_names,
        }

        with pm.Model(coords=coords) as self.model:
            # --- Priors ---
            intercept = pm.Normal("intercept", mu=0, sigma=10)
            
            # Priors for variant coefficients with informative logic
            # This demonstrates how to incorporate prior knowledge into the model.
            variant_coeffs = []
            for var_name in variant_names:
                if 'vgsc' in var_name.lower() or 'ace1' in var_name.lower():
                    print(f"Using INFORMATIVE prior for known gene: {var_name}")
                    # Prior centered on a positive effect (mu=1.0) with some uncertainty
                    beta = pm.Normal(f"beta_{var_name}", mu=1.0, sigma=0.75)
                else:
                    print(f"Using NON-INFORMATIVE prior for: {var_name}")
                    # Weakly regularizing prior centered on zero for other variants
                    beta = pm.Normal(f"beta_{var_name}", mu=0, sigma=2.5)
                variant_coeffs.append(beta)
            
            # Combine the individual priors into a single tensor
            variant_coeffs = pm.math.stack(variant_coeffs)
            
            # Priors for PC coefficients (always non-informative)
            pc_coeffs = pm.Normal("pc_coeffs", mu=0, sigma=2.5, dims="pc_predictors")

            # --- Linear Model ---
            logit_p = intercept + pm.math.dot(X_variants, variant_coeffs) + pm.math.dot(X_pcs, pc_coeffs)

            # --- Likelihood ---
            pm.Bernoulli("obs", logit_p=logit_p, observed=phenotype)

            # --- MCMC Sampling ---
            print("Starting MCMC sampling... (This may take a few minutes)")
            self.idata = pm.sample(
                draws=2000,
                tune=1500,
                chains=2,
                cores=1,  # Best for compatibility
                random_seed=self.random_seed,
                progressbar=True
            )
            print("Sampling complete.")

        # Generate a summary of the posterior for the get_params method.
        self._create_summary(variant_names)

    def _create_summary(self, variant_names):
        """Generates a summary DataFrame from the inference data."""
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

        # Create a list of the variables we want to summarize
        beta_var_names = [f"beta_{var}" for var in variant_names]
        vars_to_summarize = beta_var_names + ["pc_coeffs"]
        
        self.summary_df = az.summary(
            self.idata,
            var_names=vars_to_summarize,
            hdi_prob=0.95
        )
        
        # Clean up the index to use the original, more readable variant names
        # This is more robust than string splitting
        param_map = {f"beta_{var}": var for var in variant_names}
        param_map.update({f"pc_coeffs[{i}]": f"PC{i+1}" for i in range(len(self.model.coords['pc_predictors']))})
        
        self.summary_df.index = self.summary_df.index.map(param_map)
        self.summary_df.index.name = "parameter"


    def get_params(self) -> pd.DataFrame:
        """
        Returns a DataFrame with a summary of the model's posterior distribution.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing summary statistics for each parameter, including
            mean, standard deviation, and 95% highest density interval (HDI).
        """
        if self.summary_df is None:
            self._create_summary()

        return self.summary_df[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%']]