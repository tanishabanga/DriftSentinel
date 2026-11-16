
"""Benchmark Logistic Regression on a chronological training subset."""

import json
import time
import warnings
from pathlib import Path

import pandas as pd
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from driftsentinel.features.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/splits/train.parquet"
VALIDATION_PATH = ROOT / "data/processed/splits/validation.parquet"
OUTPUT_PATH = ROOT / "experiments/results/logistic_regression_50k_benchmark.json"

TRAIN_ROWS = 50_000
MAX_ITER = 300


def memory_gb() -> float:
    """Return current process memory in GB."""
    return psutil.Process().memory_info().rss / (1024**3)


def main() -> None:
    print("Loading chronological training subset...", flush=True)

    train_df = pd.read_parquet(TRAIN_PATH).iloc[:TRAIN_ROWS].copy()
    validation_df = pd.read_parquet(VALIDATION_PATH)

    X_train = train_df.drop(
        columns=["isFraud", "TransactionID", "TransactionDT"]
    )
    y_train = train_df["isFraud"].astype(int)

    X_validation = validation_df.drop(
        columns=["isFraud", "TransactionID", "TransactionDT"]
    )
    y_validation = validation_df["isFraud"].astype(int)

    print(f"Training rows: {len(train_df):,}", flush=True)
    print(f"Validation rows: {len(validation_df):,}", flush=True)
    print(f"Training fraud rate: {y_train.mean():.4%}", flush=True)
    print(f"Validation fraud rate: {y_validation.mean():.4%}", flush=True)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=MAX_ITER,
                    class_weight=None,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    start = time.perf_counter()
    print("\nFitting Logistic Regression...", flush=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)

    training_seconds = time.perf_counter() - start
    classifier = pipeline.named_steps["classifier"]

    print(f"Training completed: {training_seconds:.2f}s", flush=True)
    print(f"Iterations: {classifier.n_iter_.tolist()}", flush=True)
    print(f"Convergence warnings: {len(caught)}", flush=True)
    print(f"Process memory: {memory_gb():.2f} GB", flush=True)

    print("\nScoring validation set...", flush=True)
    start = time.perf_counter()

    probabilities = pipeline.predict_proba(X_validation)[:, 1]

    inference_seconds = time.perf_counter() - start

    results = {
        "experiment": "logistic_regression_50k_benchmark",
        "status": "completed",
        "model": "LogisticRegression",
        "configuration": {
            "solver": "liblinear",
            "max_iter": MAX_ITER,
            "class_weight": None,
            "random_state": 42,
        },
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
            "brier_score": float(
                brier_score_loss(y_validation, probabilities)
            ),
            "log_loss": float(
                log_loss(y_validation, probabilities)
            ),
        },
        "resources": {
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "process_memory_gb": memory_gb(),
            "iterations": classifier.n_iter_.tolist(),
            "convergence_warnings": len(caught),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    print("\nBENCHMARK RESULTS", flush=True)
    print(json.dumps(results, indent=4), flush=True)
    print(f"\nSaved to: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
