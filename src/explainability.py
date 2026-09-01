"""
Cross-architecture explainability analysis.

Reconciles feature importance across model families:
  - Linear models  : |standardized coefficients|
  - Tree ensembles : Gini (impurity-based) importance

Then aggregates the share of total importance attributable to interaction
terms (E x S, E x G, S x G) vs base terms (E, S, G) vs controversy, which is
the paper's headline structural result.
"""

import numpy as np
import pandas as pd

INTERACTION_FEATURES = ["E_x_S", "E_x_G", "S_x_G"]
BASE_FEATURES = ["E", "S", "G"]
CONTROVERSY_FEATURES = ["Controversy"]


def normalized_importance(values, feature_names) -> pd.Series:
    """Take absolute values and normalize to sum to 1."""
    v = np.abs(np.asarray(values, dtype=float))
    if v.sum() == 0:
        return pd.Series(np.zeros_like(v), index=feature_names)
    return pd.Series(v / v.sum(), index=feature_names)


def gini_importance(tree_model, feature_names) -> pd.Series:
    return normalized_importance(tree_model.feature_importances_, feature_names)


def reconcile(importances: dict) -> pd.DataFrame:
    """importances: {model_name: pd.Series over feature_names}.
    Returns a tidy table of normalized importance per feature per model."""
    return pd.DataFrame(importances)


def aggregate_shares(importance: pd.Series) -> dict:
    """Share of total importance by feature group."""
    return {
        "interactions": importance[INTERACTION_FEATURES].sum(),
        "base": importance[BASE_FEATURES].sum(),
        "controversy": importance[CONTROVERSY_FEATURES].sum(),
        "strongest_single": importance.idxmax(),
        "strongest_single_share": importance.max(),
    }
