import statsmodels.formula.api as smf
import pandas as pd
from sklearn.metrics import roc_auc_score


class LogisticRegressionGWAS:
    """
    Fits a logistic regression model. Assumes the input DataFrame is fully prepared.
    """

    def __init__(self):
        self.results = None
        self.analysis_df = None  # Store the dataframe used for fitting

    def fit(
        self,
        analysis_df: pd.DataFrame,
        variant_names: list,
        pc_names: list,
        include_interaction: bool = False,
    ):
        """
        Fits the logistic regression model.

        Parameters
        ----------
        analysis_df : pd.DataFrame
            The fully prepared data for modeling.
        variant_names : list
            List of the original variant names.
        pc_names : list
            List of the PC column names.
        include_interaction : bool, optional
            If True, includes an interaction term between the variants, by default False.
        """
        self.analysis_df = analysis_df
        variant_terms = [f"has_{name}" for name in variant_names]
        # Base formula with main effects
        formula = f"phenotype ~ {' + '.join(variant_terms)}"

        # Add the interaction term if requested
        if include_interaction and len(variant_terms) > 1:
            # The '*' in the formula tells statsmodels to include main effects AND the interaction
            # e.g., has_Vgsc * has_Ace1 becomes has_Vgsc + has_Ace1 + has_Vgsc:has_Ace1
            interaction_formula = " * ".join(variant_terms)
            formula = f"phenotype ~ {interaction_formula}"

        # Add PC terms for population structure correction
        if pc_names:
            formula += f" + {' + '.join(pc_names)}"

        print(f"\nUsing formula for Logistic Regression: {formula}\n")

        model = smf.logit(formula, data=self.analysis_df)
        self.results = model.fit(disp=0)

        return self

    def summary(self):
        """Returns the statsmodels summary of the fitted model."""
        if self.results:
            return self.results.summary()
        return "Model has not been fitted yet."

    def get_params(self) -> pd.DataFrame:
        """
        Extracts key parameters (coefficient, std err, p-value, CIs) into a DataFrame.
        """
        if not self.results:
            return pd.DataFrame()

        params = self.results.params
        stderr = self.results.bse
        pvalues = self.results.pvalues
        conf_int = self.results.conf_int()

        results_df = pd.DataFrame(
            {"coefficient": params, "std_err": stderr, "p_value": pvalues}
        )
        results_df["conf_int_lower"] = conf_int[0]
        results_df["conf_int_upper"] = conf_int[1]

        return results_df

    def get_performance_metrics(self) -> dict:
        """
        Calculates and returns performance metrics for the model.
        """
        if not self.results or self.analysis_df is None:
            return {}

        predicted_probs = self.results.predict(self.analysis_df)

        auc = roc_auc_score(self.analysis_df["phenotype"], predicted_probs)

        metrics = {
            "auc": auc,
            "log-likelihood": self.results.llf,
            "pseudo_r-squ": self.results.prsquared,
            "predicted_probabilities": predicted_probs,  # Return for plotting
        }
        return metrics
