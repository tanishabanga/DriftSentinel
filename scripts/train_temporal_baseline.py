
"""Train and save the initial temporal baseline model."""

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from driftsentinel.models.baseline import (
    build_logistic_regression_pipeline,
)

TRAIN_PATH = Path(
    "data/processed/temporal/initial_train_sample.parquet"
)

ARTIFACT_DIR = Path("artifacts/temporal")
RESULTS_DIR = Path("experiments/results")

MODEL_PATH = ARTIFACT_DIR / "baseline_model.joblib"
METADATA_PATH = RESULTS_DIR / "temporal_baseline_metadata.json"


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading initial training sample...")
    df = pd.read_parquet(TRAIN_PATH)

    y_train = df["isFraud"]
    X_train = df.drop(
        columns=["isFraud", "TransactionID", "TransactionDT"]
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Features before preprocessing: {X_train.shape[1]}")

    model = build_logistic_regression_pipeline(X_train)

    print("\nTraining Logistic Regression...")
    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_seconds = time.perf_counter() - start_time

    print(f"Training completed in {training_seconds:.2f} seconds.")

    print("\nSaving fitted model...")
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "temporal_logistic_regression_baseline",
        "model_type": "LogisticRegression",
        "training_rows": len(X_train),
        "training_fraud_count": int(y_train.sum()),
        "training_fraud_percentage": round(
            y_train.mean() * 100, 4
        ),
        "input_feature_count": X_train.shape[1],
        "training_seconds": round(training_seconds, 4),
        "random_state": 42,
        "max_iter": 1000,
        "solver": "liblinear",
        "class_weight": None,
        "artifact_path": str(MODEL_PATH),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nBASELINE MODEL METADATA")
    print(json.dumps(metadata, indent=4))

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
