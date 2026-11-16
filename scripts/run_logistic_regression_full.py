
"""Run full-data Logistic Regression fraud detection experiment."""

import json
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from driftsentinel.models.baseline import (
    EXCLUDED_COLUMNS,
    build_logistic_regression_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/splits/train.parquet"
VALIDATION_PATH = ROOT / "data/processed/splits/validation.parquet"

RESULTS_DIR = ROOT / "experiments/results"

METADATA_PATH = RESULTS_DIR / "logistic_regression_full_metadata.json"
PREDICTIONS_PATH = (
    RESULTS_DIR / "logistic_regression_full_predictions.parquet"
)

MODEL_PATH = ROOT / "artifacts/logistic_regression/fraud_pipeline.joblib"

THRESHOLD = 0.5


def memory_gb() -> float:
    """Return current process memory usage in GB."""
    return psutil.Process().memory_info().rss / (1024**3)


def save_metadata(results: dict) -> None:
    """Persist experiment metadata."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    METADATA_PATH.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 65)
    print("DRIFTSENTINEL: FULL-DATA LOGISTIC REGRESSION")
    print("=" * 65)

    results = {
        "experiment": "logistic_regression_full_data",
        "status": "running",
        "model": "LogisticRegression",
        "configuration": {
            "solver": "liblinear",
            "max_iter": 1000,
            "class_weight": None,
            "random_state": 42,
            "threshold": THRESHOLD,
        },
        "resources": {
            "initial_memory_gb": memory_gb(),
            "available_system_memory_gb": (
                psutil.virtual_memory().available / (1024**3)
            ),
        },
    }

    save_metadata(results)

    try:
        print("\nLoading training data...")
        train_df = pd.read_parquet(TRAIN_PATH)
        print(f"Training rows: {len(train_df):,}")

        print("\nLoading validation data...")
        validation_df = pd.read_parquet(VALIDATION_PATH)
        print(f"Validation rows: {len(validation_df):,}")

        X_train = train_df.drop(columns=EXCLUDED_COLUMNS)
        y_train = train_df["isFraud"].astype(int)

        X_validation = validation_df.drop(columns=EXCLUDED_COLUMNS)
        y_validation = validation_df["isFraud"].astype(int)

        pipeline = build_logistic_regression_pipeline(X_train)

        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        results["sampling"] = {
            "method": "full_chronological_partitions",
            "training_rows": len(y_train),
            "validation_rows": len(y_validation),
            "training_fraud_rate": float(y_train.mean()),
            "validation_fraud_rate": float(y_validation.mean()),
            "feature_count": X_train.shape[1],
        }

        print("\nFitting preprocessing...")
        preprocessing_start = time.perf_counter()

        X_train_transformed = preprocessor.fit_transform(X_train)

        preprocessing_seconds = (
            time.perf_counter() - preprocessing_start
        )

        print(
            f"Preprocessing completed in "
            f"{preprocessing_seconds:.2f} seconds"
        )
        print("Transformed training shape:", X_train_transformed.shape)
        print("Sparse output:", hasattr(X_train_transformed, "nnz"))
        print("Output dtype:", X_train_transformed.dtype)
        print(f"Memory: {memory_gb():.2f} GB")

        print("\nFitting Logistic Regression...")
        fitting_start = time.perf_counter()

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)

            classifier.fit(X_train_transformed, y_train)

        fitting_seconds = time.perf_counter() - fitting_start

        convergence_warnings = [
            str(item.message)
            for item in caught_warnings
            if issubclass(item.category, ConvergenceWarning)
        ]

        print(f"Classifier fitting completed in {fitting_seconds:.2f} seconds")
        print("Iterations:", classifier.n_iter_)
        print("Convergence warnings:", len(convergence_warnings))
        print(f"Memory: {memory_gb():.2f} GB")

        print("\nTransforming validation data...")
        validation_start = time.perf_counter()

        X_validation_transformed = preprocessor.transform(X_validation)

        preprocessing_inference_seconds = (
            time.perf_counter() - validation_start
        )

        print("\nGenerating validation probabilities...")
        inference_start = time.perf_counter()

        probabilities = classifier.predict_proba(
            X_validation_transformed
        )[:, 1]

        inference_seconds = time.perf_counter() - inference_start

        predictions = (probabilities >= THRESHOLD).astype(int)

        # Calculate validation metrics.
        results["metrics"] = {
            "average_precision": float(
                average_precision_score(y_validation, probabilities)
            ),
            "roc_auc": float(
                roc_auc_score(y_validation, probabilities)
            ),
            "precision": float(
                precision_score(
                    y_validation, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_validation, predictions, zero_division=0
                )
            ),
            "f1": float(
                f1_score(y_validation, predictions, zero_division=0)
            ),
            "brier_score": float(
                brier_score_loss(y_validation, probabilities)
            ),
            "log_loss": float(
                log_loss(
                    y_validation,
                    probabilities,
                    labels=[0, 1],
                )
            ),
            "confusion_matrix": confusion_matrix(
                y_validation, predictions
            ).tolist(),
        }

        results["resources"].update({
            "preprocessing_seconds": preprocessing_seconds,
            "classifier_fitting_seconds": fitting_seconds,
            "validation_preprocessing_seconds": (
                preprocessing_inference_seconds
            ),
            "prediction_seconds": inference_seconds,
            "final_memory_gb": memory_gb(),
        })

        results["training_diagnostics"] = {
            "iterations": classifier.n_iter_.tolist(),
            "convergence_warning_count": len(convergence_warnings),
            "convergence_warnings": convergence_warnings,
        }

        print("\nSaving validation predictions...")

        predictions_df = validation_df[
            ["TransactionID", "TransactionDT", "isFraud"]
        ].copy()

        predictions_df["probability"] = probabilities

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        predictions_df.to_parquet(PREDICTIONS_PATH, index=False)

        print("Saving trained pipeline...")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        results["status"] = "validation_evaluated"
        save_metadata(results)

        print("\n" + "=" * 65)
        print("LOGISTIC REGRESSION RESULTS")
        print("=" * 65)
        print(json.dumps(results, indent=4))

        print("\nSaved predictions:", PREDICTIONS_PATH)
        print("Saved model:", MODEL_PATH)
        print("Saved metadata:", METADATA_PATH)

    except Exception as error:
        results["status"] = "failed"
        results["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }

        save_metadata(results)
        raise


if __name__ == "__main__":
    main()
