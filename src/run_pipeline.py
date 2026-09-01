"""
End-to-end pipeline runner. Reproduces the paper's methodology:

  1. Load + clean + engineer features + stratified 80/20 split
  2. Train four model families (classification + regression)
  3. Cross-architecture explainability reconciliation
  4. Aggregate interaction vs base importance shares
  5. Write metrics tables (results/) and figures (figures/)

Run:  python src/run_pipeline.py --data data/esg_sample.csv
"""

import argparse
import os
import pandas as pd

from data_prep import prepare, FEATURE_NAMES
from evaluate import (
    run_classification, run_regression,
    standardized_coefficients, logistic_coefficients,
)
from explainability import (
    gini_importance, normalized_importance, reconcile, aggregate_shares,
)
import figures


def main(data_path: str):
    os.makedirs("results", exist_ok=True)

    print("\n=== 1. Data preparation ===")
    data = prepare(data_path)

    print("\n=== 2a. Classification ===")
    clf_metrics, clf_models = run_classification(data)
    print(clf_metrics.round(4))
    clf_metrics.to_csv("results/classification_metrics.csv")

    print("\n=== 2b. Regression ===")
    reg_metrics, reg_models = run_regression(data)
    print(reg_metrics.round(4))
    reg_metrics.to_csv("results/regression_metrics.csv")

    print("\n=== 3. Interpretable coefficients ===")
    logit_coefs = logistic_coefficients(clf_models["Logistic Regression"], FEATURE_NAMES)
    ridge_coefs = standardized_coefficients(reg_models["Ridge Regression"], FEATURE_NAMES)
    print("Logistic (|coef| ranked):\n", logit_coefs.round(4))
    print("Ridge standardized coefficients:\n", ridge_coefs.round(4))
    logit_coefs.to_csv("results/logistic_coefficients.csv")
    ridge_coefs.to_csv("results/ridge_coefficients.csv")

    print("\n=== 4. Cross-architecture explainability ===")
    rf_clf_imp = gini_importance(clf_models["Random Forest"], FEATURE_NAMES)
    rf_reg_imp = gini_importance(reg_models["Random Forest"], FEATURE_NAMES)
    ridge_imp = normalized_importance(ridge_coefs.reindex(FEATURE_NAMES).values, FEATURE_NAMES)
    logit_imp = normalized_importance(logit_coefs.reindex(FEATURE_NAMES).values, FEATURE_NAMES)

    table = reconcile({
        "Logistic |coef|": logit_imp,
        "Ridge |coef|": ridge_imp,
        "RF (clf) Gini": rf_clf_imp,
        "RF (reg) Gini": rf_reg_imp,
    })
    table.to_csv("results/importance_reconciliation.csv")
    print(table.round(4))

    # Consensus importance = mean across architectures
    consensus = table.mean(axis=1)
    shares = aggregate_shares(consensus)
    print("\n=== 5. Importance shares (consensus) ===")
    print(f"  Interactions : {shares['interactions']*100:.1f}%")
    print(f"  Base (E,S,G) : {shares['base']*100:.1f}%")
    print(f"  Controversy  : {shares['controversy']*100:.1f}%")
    print(f"  Strongest single predictor: {shares['strongest_single']} "
          f"({shares['strongest_single_share']*100:.1f}%)")
    pd.Series(shares).to_csv("results/importance_shares.csv")

    print("\n=== 6. Figures ===")
    figures.plot_rf_importance(rf_reg_imp)
    figures.plot_reconciliation(table)
    figures.plot_shares(shares)
    print("  Wrote figures to figures/")

    print("\nDone. Metrics in results/, figures in figures/.")
    if "sample" in data_path.lower() or "synthetic" in data_path.lower():
        print("NOTE: synthetic data -> numbers will differ from the paper.")
    else:
        print("NOTE: importance shares depend on interaction-feature "
              "construction; see README.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/esg_sample.csv")
    args = p.parse_args()
    main(args.data)
