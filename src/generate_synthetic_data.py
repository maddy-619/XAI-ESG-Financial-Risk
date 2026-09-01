"""
Generate a synthetic ESG dataset that mimics the structure of the S&P 500
ESG Risk Ratings data used in the paper. This lets the pipeline run end-to-end
without the proprietary Refinitiv-derived data.

The synthetic generator encodes the paper's central structural finding:
governance acts as a MULTIPLICATIVE conditioner on environmental and social
performance, so interaction terms (E x S, E x G, S x G) carry real signal.

NOTE: numbers produced from this synthetic file will NOT match the paper.
Replace data/esg_sample.csv with your real dataset to reproduce paper results.
"""

import argparse
import numpy as np
import pandas as pd


def generate(n_firms: int = 503, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Base sub-dimension scores (0-100 scale, lower = lower risk exposure)
    environment = rng.beta(2.5, 3.0, n_firms) * 100
    social = rng.beta(2.8, 3.2, n_firms) * 100
    governance = rng.beta(3.0, 2.5, n_firms) * 100
    controversy = rng.integers(0, 6, n_firms).astype(float)  # 0-5 controversy flag

    # Standardize for building the latent target
    def z(x):
        return (x - x.mean()) / x.std()

    zE, zS, zG = z(environment), z(social), z(governance)

    # Latent total ESG risk: interactions dominate (governance multiplies E and S),
    # matching the paper's finding that interactions ~= 64.5% of importance and
    # E x S is the single strongest predictor.
    latent = (
        0.10 * zE
        + 0.12 * zS
        + 0.09 * zG
        + 0.30 * (zE * zS)      # strongest single predictor
        + 0.18 * (zE * zG)
        + 0.16 * (zS * zG)
        + 0.05 * z(controversy)
        + rng.normal(0, 0.15, n_firms)  # small irreducible noise
    )

    # Map latent to a Total ESG Risk Score on a plausible 0-40 scale
    total = (latent - latent.min()) / (latent.max() - latent.min()) * 40

    df = pd.DataFrame(
        {
            "Symbol": [f"SYN{i:04d}" for i in range(n_firms)],
            "Environment_Score": environment.round(2),
            "Social_Score": social.round(2),
            "Governance_Score": governance.round(2),
            "Controversy_Score": controversy,
            "Total_ESG_Risk_Score": total.round(2),
        }
    )

    # Inject ~14.5% missingness across sub-dimensions so the cleaning step
    # (503 -> ~430 firms) mirrors the paper's listwise deletion.
    n_missing = int(n_firms * 0.145)
    miss_idx = rng.choice(n_firms, n_missing, replace=False)
    cols = ["Environment_Score", "Social_Score", "Governance_Score"]
    for i in miss_idx:
        c = rng.choice(cols)
        df.loc[i, c] = np.nan

    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/esg_sample.csv")
    p.add_argument("--n", type=int, default=503)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = generate(args.n, args.seed)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} synthetic firms to {args.out}")
