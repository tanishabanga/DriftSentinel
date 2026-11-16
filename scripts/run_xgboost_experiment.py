
"""Run the full-data XGBoost fraud detection experiment."""

import json
import time
from pathlib import Path

import pandas as pd
import psutil
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

from driftsentinel.models.xgboost_model import (
    build_xgboost_classifier,
    prepare_xgboost_features,
)

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/splits/train.parquet"
VALIDATION_PATH = ROOT / "data/processed/splits/validation.parquet"

RESULTS_DIR = ROOT / "experiments/results"
PREDICTIONS_PATH = RESULTS_DIR / "xgboost_full_predictions.parquet"
METADATA_PATH = RESULTS_DIR / "xgboost_full_metadata.json"

MODEL_PATH = ROOT / "artifacts/xgboost/fraud_model.json"

EXCLUDED_COLUMNS = [
    "isFraud",
    "TransactionID",
    "TransactionDT",
]

THRESHOLD = 0.5


def memory_gb() -> float:
    """Return current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def main() -> None:
    print("=" * 65)
    print("DRIFTSENTINEL: XGBOOST FULL-DATA EXPERIMENT")
    print("=" * 65)

    print(f"\nInitial process memory: {memory_gb():.2f} GB")

    # 1. Load chronological training partition.
    print("\nLoading training data...")

    train_df = pd.read_parquet(TRAIN_PATH)

    print(f"Training rows: {len(train_df):,}")
    print(f"Memory: {memory_gb():.2f} GB")

    # 2. Load chronological validation partition.
    print("\nLoading validation data...")

    validation_df = pd.read_parquet(VALIDATION_PATH)

    print(f"Validation rows: {len(validation_df):,}")
    print(f"Memory: {memory_gb():.2f} GB")

    # Preserve validation identifiers separately.
    validation_metadata = validation_df[
        ["TransactionID", "TransactionDT"]
    ].copy()

    # 3. Separate features and labels.
    X_train = train_df.drop(columns=EXCLUDED_COLUMNS)
    y_train = train_df["isFraud"].astype("int8")

    X_validation = validation_df.drop(columns=EXCLUDED_COLUMNS)
    y_validation = validation_df["isFraud"].astype("int8")

    del train_df
    del validation_df

    print("\nTraining fraud rate:", f"{y_train.mean():.4%}")
    print("Validation fraud rate:", f"{y_validation.mean():.4%}")

    # 4. Prepare numerical and native categorical features.
    print("\nPreparing XGBoost features...")

    X_train, X_validation = prepare_xgboost_features(
        X_train,
        X_validation,
    )

    print(f"Feature count: {X_train.shape[1]}")
    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_validation.shape}")
    print(f"Memory: {memory_gb():.2f} GB")

    # 5. Build model.
    model = build_xgboost_classifier()

    print("\nMODEL CONFIGURATION")
    print(model.get_params())

    # 6. Train using validation-based early stopping.
    memory_before_fit = memory_gb()

    print("\nStarting XGBoost training...")
    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=50,
    )

    training_seconds = time.perf_counter() - start_time
    memory_after_fit = memory_gb()

    print("\nTraining completed.")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Memory after fit: {memory_after_fit:.2f} GB")

    # 7. Generate validation probabilities.
    print("\nGenerating validation predictions...")

    start_time = time.perf_counter()

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    inference_seconds = time.perf_counter() - start_time

    predictions = (probabilities >= THRESHOLD).astype(int)

    # 8. Calculate validation metrics.
    results = {
        "experiment": "xgboost_full_data",
        "status": "validation_evaluated",
        "model": "XGBClassifier",
        "threshold": THRESHOLD,
        "configuration": {
            "tree_method": "hist",
            "enable_categorical": True,
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "scale_pos_weight": 1,
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": 2,
        },
        "sampling": {
            "method": "full_chronological_partitions",
            "training_rows": len(y_train),
            "validation_rows": len(y_validation),
            "training_fraud_rate": float(y_train.mean()),
            "validation_fraud_rate": float(y_validation.mean()),
            "feature_count": X_train.shape[1],
        },
        "metrics": {
            "average_precision": float(
                average_precision_score(y_validation, probabilities)
            ),
            "roc_auc": float(
                roc_auc_score(y_validation, probabilities)
            ),
            "precision": float(
                precision_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                )
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
                y_validation,
                predictions,
            ).tolist(),
        },
        "resources": {
            "memory_before_fit_gb": memory_before_fit,
            "memory_after_fit_gb": memory_after_fit,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
        },
        "early_stopping": {
            "best_iteration": int(model.best_iteration),
            "best_score": float(model.best_score),
        },
    }

    # 9. Save validation predictions.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = validation_metadata.copy()
    predictions_df["isFraud"] = y_validation.to_numpy()
    predictions_df["probability"] = probabilities

    predictions_df.to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )

    # 10. Save trained model.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_PATH))

    # 11. Save experiment metadata.
    METADATA_PATH.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("XGBOOST EXPERIMENT RESULTS")
    print("=" * 65)

    print(json.dumps(results, indent=4))

    print("\nSaved predictions:", PREDICTIONS_PATH)
    print("Saved model:", MODEL_PATH)
    print("Saved metadata:", METADATA_PATH)


if __name__ == "__main__":
    main()
