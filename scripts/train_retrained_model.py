
"""Train a Logistic Regression model using newly available data."""

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from driftsentinel.models.baseline import (
    build_logistic_regression_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "temporal"
ARTIFACT_DIR = ROOT / "artifacts" / "temporal"
RESULTS_DIR = ROOT / "experiments" / "results"

INITIAL_TRAIN_PATH = DATA_DIR / "initial_train_sample.parquet"
ADAPTATION_TRAIN_PATH = DATA_DIR / "adaptation_train.parquet"

MODEL_PATH = ARTIFACT_DIR / "retrained_model.joblib"
METADATA_PATH = RESULTS_DIR / "retrained_model_metadata.json"

RANDOM_STATE = 42


def load_training_data() -> pd.DataFrame:
    """Load and validate the original and adaptation training data."""

    initial = pd.read_parquet(INITIAL_TRAIN_PATH)
    adaptation = pd.read_parquet(ADAPTATION_TRAIN_PATH)

    required_columns = {
        "TransactionID",
        "TransactionDT",
        "isFraud",
    }

    for name, df in [
        ("initial training", initial),
        ("adaptation training", adaptation),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: {missing}"
            )

    if initial["TransactionID"].duplicated().any():
        raise ValueError("Duplicate IDs in initial training data.")

    if adaptation["TransactionID"].duplicated().any():
        raise ValueError("Duplicate IDs in adaptation training data.")

    overlap = set(initial["TransactionID"]) & set(
        adaptation["TransactionID"]
    )

    if overlap:
        raise ValueError(
            f"Training partitions overlap: {len(overlap)} IDs."
        )

    combined = pd.concat(
        [initial, adaptation],
        ignore_index=True,
    )

    combined = combined.sort_values(
        "TransactionDT",
        kind="mergesort",
    ).reset_index(drop=True)

    if combined["isFraud"].nunique() != 2:
        raise ValueError("Training data must contain both classes.")

    return combined


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training partitions...")

    df = load_training_data()

    y_train = df["isFraud"].astype(int)

    X_train = df.drop(
        columns=["isFraud", "TransactionID", "TransactionDT"]
    )

    print(f"Combined training rows: {len(X_train):,}")
    print(f"Fraud cases: {int(y_train.sum()):,}")
    print(f"Fraud percentage: {y_train.mean() * 100:.4f}%")
    print(f"Input features: {X_train.shape[1]}")

    model = build_logistic_regression_pipeline(X_train)

    print("\nTraining retrained Logistic Regression...")

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start_time

    print(f"Training completed in {training_seconds:.2f} seconds.")

    print("\nSaving retrained model...")

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "temporal_retrained_logistic_regression",
        "model_type": "LogisticRegression",
        "training_rows": len(X_train),
        "training_fraud_count": int(y_train.sum()),
        "training_fraud_percentage": round(
            y_train.mean() * 100, 4
        ),
        "input_feature_count": X_train.shape[1],
        "training_seconds": round(training_seconds, 4),
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
        "solver": "liblinear",
        "class_weight": None,
       "initial_training_rows": len(
            pd.read_parquet(INITIAL_TRAIN_PATH)
        ),
        "adaptation_training_rows": len(
            pd.read_parquet(ADAPTATION_TRAIN_PATH)
        ),
        "training_period_min": int(df["TransactionDT"].min()),
        "training_period_max": int(df["TransactionDT"].max()),
        "artifact_path": str(MODEL_PATH),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nRETRAINED MODEL METADATA")
    print(json.dumps(metadata, indent=4))

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
