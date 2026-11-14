
"""Run a resource-monitored Logistic Regression pilot."""

import json
import time
from pathlib import Path

import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from driftsentinel.data.sampling import read_parquet_sample
from driftsentinel.features.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/splits/train.parquet"
VALIDATION_PATH = ROOT / "data/processed/splits/validation.parquet"
OUTPUT_PATH = ROOT / "experiments/results/logistic_regression_pilot.json"

TRAIN_ROWS = 20_000
VALIDATION_ROWS = 5_000
THRESHOLD = 0.5

EXCLUDED_COLUMNS = [
    "isFraud",
    "TransactionID",
    "TransactionDT",
]


def memory_gb() -> float:
    """Return current process resident memory in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def main() -> None:
    print("Starting DriftSentinel Logistic Regression pilot")
    print(f"Initial process memory: {memory_gb():.2f} GB")

    # 1. Load controlled chronological samples.
    train_df = read_parquet_sample(TRAIN_PATH, TRAIN_ROWS)
    print(f"Training sample loaded: {len(train_df):,} rows")
    print(f"Process memory: {memory_gb():.2f} GB")

    validation_df = read_parquet_sample(
        VALIDATION_PATH,
        VALIDATION_ROWS,
    )
    print(f"Validation sample loaded: {len(validation_df):,} rows")
    print(f"Process memory: {memory_gb():.2f} GB")

    X_train = train_df.drop(columns=EXCLUDED_COLUMNS)
    y_train = train_df["isFraud"].astype(int)

    X_validation = validation_df.drop(columns=EXCLUDED_COLUMNS)
    y_validation = validation_df["isFraud"].astype(int)

    print(f"Training fraud rate: {y_train.mean():.4%}")
    print(f"Validation fraud rate: {y_validation.mean():.4%}")

    # 2. Build the complete preprocessing + model pipeline.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=None,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    # 3. Fit while measuring resource consumption.
    memory_before_fit = memory_gb()
    start = time.perf_counter()

    print("\nFitting Logistic Regression...")
    pipeline.fit(X_train, y_train)

    training_seconds = time.perf_counter() - start
    memory_after_fit = memory_gb()

    print(f"Training completed in {training_seconds:.2f} seconds")
    print(f"Process memory after fit: {memory_after_fit:.2f} GB")

    # 4. Predict validation probabilities.
    print("\nGenerating validation predictions...")
    start = time.perf_counter()

    probabilities = pipeline.predict_proba(X_validation)[:, 1]

    inference_seconds = time.perf_counter() - start
    predictions = (probabilities >= THRESHOLD).astype(int)

    # Save validation predictions for threshold experiments
    predictions_df = validation_df[
        ["TransactionID", "TransactionDT"]
    ].copy()

    predictions_df["isFraud"] = y_validation.to_numpy()
    predictions_df["probability"] = probabilities

    predictions_path = Path(
        "experiments/results/logistic_regression_pilot_predictions.parquet"
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(predictions_path, index=False)

    print(f"Validation predictions saved to: {predictions_path}")

    # 5. Evaluate.
    results = {
        "experiment": "logistic_regression_pilot",
        "status": "pilot_not_final",
        "model": "LogisticRegression",
        "solver": "liblinear",
        "class_weight": None,
        "threshold": THRESHOLD,
        "sampling": {
            "method": "chronological_prefix",
            "training_rows": len(train_df),
            "validation_rows": len(validation_df),
            "training_fraud_rate": float(y_train.mean()),
            "validation_fraud_rate": float(y_validation.mean()),
        },
        "metrics": {
            "average_precision": float(
                average_precision_score(y_validation, probabilities)
            ),
            "roc_auc": float(
                roc_auc_score(y_validation, probabilities)
            ),
            "precision": float(
                precision_score(y_validation, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_validation, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y_validation, predictions, zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                y_validation, predictions
            ).tolist(),
        },
        "resources": {
            "memory_before_fit_gb": memory_before_fit,
            "memory_after_fit_gb": memory_after_fit,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
        },
    }

    # 6. Persist results.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    print("\nPILOT RESULTS")
    print(json.dumps(results, indent=4))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
