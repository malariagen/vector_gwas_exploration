import pandas as pd
import statsmodels.formula.api as smf
import warnings
from typing import List


class MixedEffectsGWAS:
    """
    Fits a Linear Mixed-Effects Model (LMM) for association testing.

    This model is a robust alternative to standard logistic regression, especially
    when the data contains non-independent samples (e.g., individuals from the
    same location). It accounts for this structure by treating a specified
    grouping variable as a random effect.

    While technically an LMM (for continuous outcomes), it is often used as a
    computationally stable and powerful approximation for binary trait GWAS
    (like 0/1 phenotypes) in large datasets, where a full Generalized LMM (GLMM)
    might be slow or fail to converge.
    """

    def __init__(self, grouping_variable: str = "country"):
        """
        Initializes the model.

        Parameters
        ----------
        grouping_variable : str, optional
            The column name in the analysis DataFrame to use for the random effects
            (e.g., "country", "sample_set"). Defaults to "country".
        """
        self.results = None
        self.analysis_df = None
        self.grouping_variable = grouping_variable

    def fit(
        self,
        analysis_df: pd.DataFrame,
        variant_names: List[str],
        pc_names: List[str],
        include_interaction: bool = False,
    ):
        """
        Fits the Linear Mixed-Effects Model.

        Parameters
        ----------
        analysis_df : pd.DataFrame
            The fully prepared data from AnalysisHelper. It MUST contain a column
            with the name specified in `self.grouping_variable`.
        variant_names : list
            List of the original variant names (e.g., ['Vgsc_L995F', 'Ace1_G280S']).
        pc_names : list
            List of the PC column names (e.g., ['PC1', 'PC2']).
        include_interaction : bool, optional
            If True, includes an interaction term between the variants. Defaults to False.
        """
        self.analysis_df = analysis_df

        if self.grouping_variable not in self.analysis_df.columns:
            raise ValueError(
                f"Grouping variable '{self.grouping_variable}' not found in the "
                "analysis DataFrame. Please ensure this column is present."
            )

        # Build the formula for the fixed effects (variant + PCs)
        variant_terms = [f"has_{name}" for name in variant_names]

        if include_interaction and len(variant_terms) > 1:
            fixed_effects_formula = " * ".join(variant_terms)
        else:
            fixed_effects_formula = " + ".join(variant_terms)

        if pc_names:
            fixed_effects_formula += f" + {' + '.join(pc_names)}"

        formula = f"phenotype ~ {fixed_effects_formula}"

        print(f"\nUsing formula for Mixed-Effects Model: {formula}")
        print(f"Grouping by random effect: '{self.grouping_variable}'\n")

        try:
            # Use `mixedlm` which fits a Linear Mixed Model. This is a common and robust
            # approach for binary traits in GWAS.
            model = smf.mixedlm(
                formula,
                data=self.analysis_df,
                groups=self.analysis_df[self.grouping_variable],
            )

            self.results = model.fit()
        except Exception as e:
            warnings.warn(f"Mixed-effects model failed to fit for {variant_terms}: {e}")
            self.results = None

        return self

    def get_params(self) -> pd.DataFrame:
        """
        Extracts key parameters for the fixed effects into a DataFrame.
        This method directly accesses model result attributes, which is more
        robust than parsing the summary table.
        """
        if not self.results:
            return pd.DataFrame()

        params = self.results.params
        stderr = self.results.bse
        pvalues = self.results.pvalues
        conf_int = self.results.conf_int()

        results_df = pd.DataFrame(
            {
                "coefficient": params,
                "std_err": stderr,
                "p_value": pvalues,
                "conf_int_lower": conf_int.iloc[:, 0],
                "conf_int_upper": conf_int.iloc[:, 1],
            }
        )

        # The random effect variance is stored separately
        # We can add it to the DataFrame for completeness
        group_var_name = f"{self.grouping_variable} Var"
        results_df.loc[group_var_name, "coefficient"] = self.results.cov_re.iloc[0, 0]

        return results_df

    def summary(self):
        """Returns the statsmodels summary of the fitted model."""
        if self.results:
            return self.results.summary()
        return "Model has not been fitted or failed to fit."
