import pandas as pd
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

    def fit(
        self,
        analysis_df: pd.DataFrame,
        variant_names: list,
        pc_names: list,
        use_informative_priors: bool = True,
        include_interaction: bool = False,
        chains: int = 2,
        cores: int = 1,
    ):
        """
        Fits the Bayesian model to the provided data using MCMC sampling.
        Includes a numerical stability fix for datasets with separation issues.

        Parameters
        ----------
        analysis_df : pd.DataFrame
            DataFrame containing the phenotype, variant data, and PCs.
        variant_names : list
            List of column names for the genetic variants.
        pc_names : list
            List of column names for the principal components.
        use_informative_priors : bool, optional
            If True, applies priors centered on a positive effect for known
            resistance genes. Defaults to True.
        include_interaction : bool, optional
            If True and there are exactly two variants, includes an interaction
            term between them. Defaults to False.
        chains : int, optional
            The number of MCMC chains to run. Defaults to 2.
        cores : int, optional
            The number of CPU cores to use for parallel sampling. Defaults to 1.
            For best performance, set this equal to `chains`.
        """
        phenotype = analysis_df["phenotype"].values
        X_variants = analysis_df[variant_names].values
        X_pcs = analysis_df[pc_names].values

        coords = {
            "variant_predictors": variant_names,
            "pc_predictors": pc_names,
        }

        with pm.Model(coords=coords) as self.model:
            intercept = pm.Normal("intercept", mu=0, sigma=10)

            # --- Main Effects for Variants ---
            variant_coeffs_list = []
            for var_name in variant_names:
                if use_informative_priors and (
                    "vgsc" in var_name.lower() or "ace1" in var_name.lower()
                ):
                    print(f"Using INFORMATIVE prior for known gene: {var_name}")
                    beta = pm.Normal(f"beta_{var_name}", mu=1.0, sigma=0.75)
                else:
                    print(f"Using NON-INFORMATIVE prior for: {var_name}")
                    # --- CHANGE 1: STRONGER PRIOR ---
                    # Changed sigma from 2.5 to 1.0 to regularize the model and
                    # improve stability, preventing coefficients from becoming too large.
                    beta = pm.Normal(f"beta_{var_name}", mu=0, sigma=1.0)
                variant_coeffs_list.append(beta)

            variant_coeffs = pm.math.stack(variant_coeffs_list)

            # --- Covariates ---
            # --- CHANGE 2: STRONGER PRIOR ---
            # Also applied the stronger prior to PC coefficients for consistency.
            pc_coeffs = pm.Normal("pc_coeffs", mu=0, sigma=1.0, dims="pc_predictors")

            # --- Linear Model (start with main effects) ---
            logit_p = (
                intercept
                + pm.math.dot(X_variants, variant_coeffs)
                + pm.math.dot(X_pcs, pc_coeffs)
            )

            # --- Interaction Term (Optional) ---
            interaction_term_name = None
            if include_interaction and len(variant_names) == 2:
                interaction_term_name = f"beta_{variant_names[0]}:{variant_names[1]}"
                print(f"Including INTERACTION term: {interaction_term_name}")

                # Applied stronger prior to the interaction term as well.
                beta_interaction = pm.Normal(interaction_term_name, mu=0, sigma=1.0)

                X_interaction = X_variants[:, 0] * X_variants[:, 1]
                logit_p += beta_interaction * X_interaction

            # Instead of passing `logit_p` directly to the Bernoulli likelihood,
            # we calculate the probability `p` and clip it to prevent it from
            # becoming exactly 0 or 1. This avoids log(0) errors and fixes the
            # "division by zero" warning, allowing the NUTS sampler to work efficiently.
            p = pm.math.invlogit(logit_p)
            epsilon = 1e-6
            p = pm.math.clip(p, epsilon, 1 - epsilon)
            pm.Bernoulli("obs", p=p, observed=phenotype)

            # --- MCMC Sampling ---
            # The `chains` and `cores` parameters are now passed in from the
            # method signature, making the function more flexible for the user.
            print(f"Starting MCMC sampling with {chains} chains on {cores} cores...")
            self.idata = pm.sample(
                draws=2000,
                tune=1500,
                chains=chains,
                cores=cores,
                random_seed=self.random_seed,
                progressbar=True,
            )
            print("Sampling complete.")

        self._create_summary(variant_names, interaction_term_name)

    def _create_summary(self, variant_names, interaction_term_name=None):
        """Generates a summary DataFrame from the inference data."""
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

        beta_var_names = [f"beta_{var}" for var in variant_names]
        vars_to_summarize = beta_var_names + ["pc_coeffs"]
        if interaction_term_name:
            vars_to_summarize.append(interaction_term_name)

        self.summary_df = az.summary(
            self.idata, var_names=vars_to_summarize, hdi_prob=0.95
        )

        param_map = {f"beta_{var}": var for var in variant_names}
        param_map.update(
            {
                f"pc_coeffs[{i}]": f"PC{i+1}"
                for i in range(len(self.model.coords["pc_predictors"]))
            }
        )
        if interaction_term_name:
            param_map[interaction_term_name] = interaction_term_name.replace(
                "beta_", ""
            )

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

        return self.summary_df[["mean", "sd", "hdi_2.5%", "hdi_97.5%"]]
