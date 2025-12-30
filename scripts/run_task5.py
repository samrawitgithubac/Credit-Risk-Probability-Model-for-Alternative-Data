"""
Convenience runner for Task 5 (Training + MLflow logging).

Usage (from repo root):
  python scripts/run_task5.py

Then open MLflow UI:
  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from src.train import TrainConfig, train_and_log_models  # noqa: WPS433

    cfg = TrainConfig(
        data_path="data/processed/processed_with_target.csv",
        experiment_name="credit-risk-task5",
        test_size=0.2,
        random_state=42,
    )
    best = train_and_log_models(cfg)

    print("\nBest model:", best["model_name"])
    print("ROC-AUC:", best["roc_auc"])
    print("Run ID:", best["run_id"])
    if best.get("registered_model"):
        print("Registered model:", best["registered_model"])
    if best.get("registered_model_error"):
        print("Model registry error:", best["registered_model_error"])

    print("\nTo view runs:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


