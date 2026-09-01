"""
Data loading, cleaning, and feature engineering.

Reproduces the paper's feature space:
  3 base features       : Environment (E), Social (S), Governance (G)
  3 interaction features: E x S, E x G, S x G
  1 controversy feature : Controversy_Score
  -> 7 engineered features total.

Cleaning: listwise deletion of firms missing any of E/S/G (503 -> ~430),
then standardization (z-score) fit on the training split only.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Column names as they appear in the Kaggle "S&P 500 ESG Risk Ratings" dataset
# (pritish509). Renamed on load to the short internal names used downstream.
COLUMN_MAP = {
    "Environment Risk Score": "Environment_Score",
    "Social Risk Score": "Social_Score",
    "Governance Risk Score": "Governance_Score",
    "Controversy Score": "Controversy_Score",
    "Total ESG Risk score": "Total_ESG_Risk_Score",
    "Symbol": "Symbol",
}

BASE_COLS = ["Environment_Score", "Social_Score", "Governance_Score"]
TARGET_COL = "Total_ESG_Risk_Score"
CONTROVERSY_COL = "Controversy_Score"

FEATURE_NAMES = [
    "E", "S", "G",
    "E_x_S", "E_x_G", "S_x_G",
    "Controversy",
]


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, standardize column names, and drop firms missing any base
    E/S/G score (listwise). On the Kaggle dataset this takes 503 -> 430."""
    df = pd.read_csv(path)

    # Rename dataset columns to internal short names if present.
    present = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=present)

    before = len(df)
    df = df.dropna(subset=BASE_COLS).reset_index(drop=True)
    after = len(df)
    print(f"Cleaning: {before} -> {after} firms "
          f"({(before - after) / before * 100:.1f}% removed)")

    # A minority of firms lack a Controversy Score; impute with 0 (no recorded
    # controversy) so the feature is defined for all retained firms.
    if CONTROVERSY_COL in df.columns:
        df[CONTROVERSY_COL] = df[CONTROVERSY_COL].fillna(0.0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 7-feature matrix. Interactions are formed on standardized
    bases so they are centered; the full matrix is standardized downstream."""
    E = df["Environment_Score"].to_numpy()
    S = df["Social_Score"].to_numpy()
    G = df["Governance_Score"].to_numpy()
    C = df.get(CONTROVERSY_COL, pd.Series(np.zeros(len(df)))).to_numpy()

    def z(x):
        return (x - x.mean()) / x.std()

    zE, zS, zG = z(E), z(S), z(G)

    X = pd.DataFrame({
        "E": E,
        "S": S,
        "G": G,
        "E_x_S": zE * zS,
        "E_x_G": zE * zG,
        "S_x_G": zS * zG,
        "Controversy": C,
    })
    return X


def make_binary_target(y: pd.Series) -> pd.Series:
    """Median split into High-ESG-risk (1) vs Low-ESG-risk (0) for the
    classification task."""
    return (y > y.median()).astype(int)


def prepare(path: str, test_size: float = 0.20, seed: int = 42):
    """Full pipeline: load -> clean -> engineer -> stratified split -> scale.

    Returns a dict with train/test matrices for both the regression target
    (continuous risk score) and the classification target (binary), plus the
    fitted scaler and feature names.
    """
    df = load_and_clean(path)
    X = engineer_features(df)
    y_cont = df[TARGET_COL]
    y_bin = make_binary_target(y_cont)

    X_tr, X_te, yc_tr, yc_te, yb_tr, yb_te = train_test_split(
        X, y_cont, y_bin,
        test_size=test_size,
        random_state=seed,
        stratify=y_bin,
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=FEATURE_NAMES, index=X_tr.index)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=FEATURE_NAMES, index=X_te.index)

    print(f"Split: train={len(X_tr)}  test={len(X_te)}  features={len(FEATURE_NAMES)}")

    return {
        "X_train": X_tr_s, "X_test": X_te_s,
        "y_cont_train": yc_tr, "y_cont_test": yc_te,
        "y_bin_train": yb_tr, "y_bin_test": yb_te,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
    }
