"""
Inference helper for Task 5.

Loads a model logged by MLflow and generates risk probabilities on a provided CSV.

Typical use:
  python -m src.predict --model-uri runs:/<run_id>/model --data data/processed/processed_with_target.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow.sklearn
import pandas as pd


TARGET_COL = "is_high_risk"


def load_model(model_uri: str):
    # We log sklearn Pipelines in src/train.py, so load via sklearn flavor.
    return mlflow.sklearn.load_model(model_uri)


def predict_proba(model, df: pd.DataFrame) -> pd.Series:
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=df.index, name="risk_probability")
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="risk_probability").astype(float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict risk probability using an MLflow model.")
    parser.add_argument("--model-uri", required=True, help="MLflow model URI (e.g., runs:/<run_id>/model)")
    parser.add_argument("--data", required=True, help="CSV path containing features (and optional is_high_risk)")
    parser.add_argument("--out", default="data/processed/predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    model = load_model(args.model_uri)
    preds = predict_proba(model, df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    out_df["risk_probability"] = preds
    out_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to: {out_path}")

