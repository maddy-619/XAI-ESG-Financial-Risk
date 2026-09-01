"""
The four model families spanning the interpretability continuum:

  Interpretable (coefficient-level transparency):
    - Logistic Regression  (classification)
    - Ridge Regression     (continuous risk-score)

  Complex (black-box):
    - Random Forest        (both tasks)
    - Feed-Forward Neural Network / MLP (both tasks)
"""

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

SEED = 42


def get_classifiers():
    """Models capable of producing class probabilities for the binary task."""
    return {
        "Logistic Regression": LogisticRegression(
            penalty="l2", C=1.0, max_iter=5000, random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=None, random_state=SEED, n_jobs=-1
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu",
            max_iter=2000, random_state=SEED
        ),
    }


def get_regressors():
    """Models for continuous Total ESG Risk Score prediction."""
    return {
        "Ridge Regression": Ridge(alpha=1.0, random_state=SEED),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=None, random_state=SEED, n_jobs=-1
        ),
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu",
            max_iter=2000, random_state=SEED
        ),
    }


# Which models are inherently interpretable (for reporting / figure styling)
INTERPRETABLE = {"Logistic Regression", "Ridge Regression"}
COMPLEX = {"Random Forest", "Neural Network (MLP)"}
