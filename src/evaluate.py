"""
Train and evaluate all model families on both tasks.

Classification metrics : Accuracy, Macro-F1, AUC-ROC
Regression metrics     : R^2, MAE (in risk-score points)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error,
)

from models import get_classifiers, get_regressors


def run_classification(data: dict) -> pd.DataFrame:
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte = data["y_bin_train"], data["y_bin_test"]

    rows = []
    fitted = {}
    for name, model in get_classifiers().items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, pred),
            "Macro_F1": f1_score(yte, pred, average="macro"),
            "AUC_ROC": roc_auc_score(yte, proba),
        })
    return pd.DataFrame(rows).set_index("Model"), fitted


def run_regression(data: dict) -> pd.DataFrame:
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte = data["y_cont_train"], data["y_cont_test"]

    rows = []
    fitted = {}
    for name, model in get_regressors().items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        pred = model.predict(Xte)
        rows.append({
            "Model": name,
            "R2": r2_score(yte, pred),
            "MAE": mean_absolute_error(yte, pred),
        })
    return pd.DataFrame(rows).set_index("Model"), fitted


def standardized_coefficients(ridge_model, feature_names) -> pd.Series:
    """Ridge coefficients on standardized features == standardized coefficients,
    directly interpretable as relative feature effect sizes."""
    return pd.Series(ridge_model.coef_, index=feature_names).sort_values(
        key=np.abs, ascending=False
    )


def logistic_coefficients(logit_model, feature_names) -> pd.Series:
    return pd.Series(logit_model.coef_[0], index=feature_names).sort_values(
        key=np.abs, ascending=False
    )
