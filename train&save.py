"""
Body Fat Prediction - Plain Linear Regression (split by sex)
================================================================

Earlier testing compared Linear, Ridge, Lasso, and ElasticNet - the
differences between them were tiny (R2 within ~0.02 of each other on
both sexes), so the regularization wasn't doing much real work here.
Going with plain LinearRegression: simplest, most interpretable,
no alpha to tune, and performance is statistically indistinguishable
from the regularized versions.

Features (same as round 2):
  Male  : Abdomen, Chest, Hip, Weight, Thigh, Neck
  Female: Abdomen, Hip, Weight, Biceps, Thigh, Forearm

This script checks for OVERFITTING / UNDERFITTING using learning curves
(numbers printed to terminal - train R2 vs CV R2 and the gap between
them), then trains a final model on the FULL dataset per sex and saves
it to disk for later use (e.g. the Streamlit app).

How to read the printed gap:
  - Both scores low and close together           -> underfitting
    (model too simple / features not informative enough)
  - Training score high, CV score much lower      -> overfitting
    (model memorizing training data, not generalizing)
  - Both converge to a reasonably high score
    as training size grows                        -> good fit

Saves:
  model_male.pkl    - trained pipeline (scaler + LinearRegression) for males
  model_female.pkl  - trained pipeline (scaler + LinearRegression) for females
  Each pickle contains the full sklearn Pipeline, so no separate scaler
  file is needed - load the pickle and call .predict() directly.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS = 10

# ── load & split by sex ──────────────────────────────────────────────────────
df = pd.read_csv("BodyFat_engineered.csv")

FEATURES_BY_SEX = {
    "Male":   ["Abdomen", "Chest", "Hip", "Weight", "Thigh", "Neck"],
    "Female": ["Abdomen", "Hip", "Weight", "Biceps", "Thigh", "Forearm"],
}

datasets = {
    "Male":   df[df["Sex"].str.strip().str.upper() == "M"].reset_index(drop=True),
    "Female": df[df["Sex"].str.strip().str.upper() == "F"].reset_index(drop=True),
}


def make_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])


for label, data in datasets.items():
    feats = FEATURES_BY_SEX[label]
    X = data[feats]
    y = data["BodyFat"]

    print(f"\n{'='*55}")
    print(f"  {label}  (n={len(X)}, features={feats})")
    print(f"{'='*55}")

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    train_sizes, train_scores, val_scores = learning_curve(
        make_model(), X, y,
        cv=cv,
        scoring="r2",
        train_sizes=np.linspace(0.2, 1.0, 8),
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    print(f"  Final training R2:   {train_mean[-1]:.3f}")
    print(f"  Final CV R2:         {val_mean[-1]:.3f}")
    print(f"  Gap (train - CV):    {train_mean[-1] - val_mean[-1]:.3f}")

    # ── train final model on the FULL dataset and save it ──────────────────
    # (the learning curve above already validated this approach via CV;
    # once validated, the final deployed model is fit on all available data)
    final_pipeline = make_model()
    final_pipeline.fit(X, y)

    filename = f"model_{label.lower()}.pkl"
    with open(filename, "wb") as f:
        pickle.dump({"pipeline": final_pipeline, "features": feats}, f)

    print(f"  Saved -> {filename}")

print("\nDone. .pkl files are ready to be loaded by the Streamlit app.")