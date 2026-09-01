"""
Generate a subset of the paper's figures from computed results.
All figures are written to figures/ as PNGs at 300 dpi.

Included here (the ones reproducible from the core pipeline):
  - feature_importance_rf.png   (Fig. 5 analogue)
  - importance_reconciliation.png (Fig. 4 / Fig. 6 analogue)
  - importance_shares.png       (Fig. 7 analogue)

Figures requiring the full experimental apparatus (reliability diagrams,
density plots, correlation matrices) can be added similarly.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def plot_rf_importance(importance: pd.Series, fname="feature_importance_rf.png"):
    imp = importance.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#2A7F7F")
    ax.set_xlabel("Normalized Gini importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)


def plot_reconciliation(table: pd.DataFrame, fname="importance_reconciliation.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table.plot(kind="bar", ax=ax)
    ax.set_ylabel("Normalized importance")
    ax.set_title("Cross-architecture feature-importance reconciliation")
    ax.legend(title="Model", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)


def plot_shares(shares: dict, fname="importance_shares.png"):
    labels = ["Interactions", "Base (E,S,G)", "Controversy"]
    vals = [shares["interactions"], shares["base"], shares["controversy"]]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(vals, labels=labels, autopct="%1.1f%%",
           colors=["#1F3A5F", "#2A7F7F", "#B8860B"])
    ax.set_title("Share of total feature importance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)
