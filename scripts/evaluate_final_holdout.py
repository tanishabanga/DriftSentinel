from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from driftsentinel.calibration.metrics import evaluate_calibration


ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    ROOT / "experiments/results/final_holdout_predictions/"
    "final_holdout_probabilities.parquet"
)
PREDICTION_METADATA_PATH = (
    ROOT / "experiments/results/final_holdout_predictions/"
    "final_holdout_metadata.json"
)
TEST_PATH = ROOT / "data/processed/splits/test.parquet"

OUTPUT_DIR = ROOT / "experiments/results/final_holdout_evaluation"
METRICS_PATH = OUTPUT_DIR / "final_holdout_metrics.csv"
METADATA_PATH = OUTPUT_DIR / "final_holdout_evaluation_metadata.json"

EXPECTED_HASHES = {
    "final_holdout_probabilities.parquet":
        "C0ACF1289CCE7733BDFAF512C97FA3A280244D4B80667D6EED0B91C7CD585496",
    "final_holdout_metadata.json":
        "1F4A4F97DC2729185C53B7E45BA090B31C963D596E6E185F7857FB04CC017040",
}

ID_COLUMNS = ["TransactionID", "TransactionDT"]

EXPECTED_STRATEGIES = [
    "frozen_uncalibrated",
    "frozen_static_sigmoid",
    "frozen_static_isotonic",
    "frozen_adaptive_sigmoid",
    "frozen_adaptive_isotonic",
    "retrained_uncalibrated",
    "retrained_sigmoid",
    "retrained_isotonic",
    "recent_only_uncalibrated",
    "recent_only_sigmoid",
    "recent_only_isotonic",
    "matched_size_uncalibrated",
    "matched_size_sigmoid",
    "matched_size_isotonic",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    if METRICS_PATH.exists() or METADATA_PATH.exists():
        raise FileExistsError(
            "Final evaluation output already exists. "
            "Do not overwrite existing results."
        )

    for path in [PREDICTION_PATH, PREDICTION_METADATA_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Verifying frozen prediction artifact hashes...")
    actual_prediction_hash = sha256(PREDICTION_PATH)
    actual_metadata_hash = sha256(PREDICTION_METADATA_PATH)

    if actual_prediction_hash != EXPECTED_HASHES[
        "final_holdout_probabilities.parquet"
    ]:
        raise ValueError("Prediction artifact hash mismatch.")

    if actual_metadata_hash != EXPECTED_HASHES[
        "final_holdout_metadata.json"
    ]:
        raise ValueError("Prediction metadata hash mismatch.")

    print("Both artifact hashes verified.")

    print("Loading frozen predictions...")
    predictions = pd.read_parquet(PREDICTION_PATH)

    missing_ids = set(ID_COLUMNS) - set(predictions.columns)
    if missing_ids:
        raise ValueError(f"Prediction file missing columns: {missing_ids}")

    strategy_columns = [
        column for column in predictions.columns
        if column not in ID_COLUMNS
    ]

    if set(strategy_columns) != set(EXPECTED_STRATEGIES):
        raise ValueError("Prediction strategy columns do not match protocol.")

    if len(predictions) != 88580:
        raise ValueError(
            f"Expected 88,580 predictions, found {len(predictions)}."
        )

    if predictions["TransactionID"].duplicated().any():
        raise ValueError("Duplicate IDs in frozen predictions.")

    if predictions["TransactionDT"].isna().any():
        raise ValueError("Missing timestamps in frozen predictions.")

    for column in EXPECTED_STRATEGIES:
        values = predictions[column].to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
        ):
            raise ValueError(f"Invalid probabilities in {column}.")

    print("Frozen predictions validated.")

    # This is the first operation in this script that reads final
    # holdout target labels. Only IDs, timestamps, and the target
    # are loaded; model input features are not loaded.
    print("Loading final holdout labels for evaluation...")
    test_labels = pd.read_parquet(
        TEST_PATH,
        columns=["TransactionID", "TransactionDT", "isFraud"],
    )

    if test_labels["TransactionID"].duplicated().any():
        raise ValueError("Duplicate IDs in final test partition.")

    if len(test_labels) != len(predictions):
        raise ValueError(
            "Prediction and test-label row counts do not match."
        )

    for column in ID_COLUMNS:
        if not predictions[column].reset_index(drop=True).equals(
            test_labels[column].reset_index(drop=True)
        ):
            raise ValueError(
                f"Prediction/test alignment failed for {column}."
            )

    y_true = test_labels["isFraud"].to_numpy()

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Final test labels are not binary.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("Final test must contain both classes.")

    print("Transaction alignment verified.")
    print(f"Final holdout rows: {len(test_labels):,}")
    print(f"Fraud transactions: {int(y_true.sum()):,}")
    print(f"Fraud prevalence: {y_true.mean():.6%}")

    print("Calculating final holdout metrics...")
    results = []

    for strategy in EXPECTED_STRATEGIES:
        probabilities = predictions[strategy].to_numpy(dtype=float)

        metrics = evaluate_calibration(y_true, probabilities)
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )

        results.append({
            "strategy": strategy,
            **metrics,
        })

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(METRICS_PATH, index=False)

    metadata = {
        "experiment": "final_chronological_holdout_evaluation",
        "prediction_stage": "frozen_predictions_evaluated",
        "test_target_loaded": True,
        "test_target_used_for_metrics_only": True,
        "models_retrained": False,
        "calibrators_refitted": False,
        "test_rows": int(len(test_labels)),
        "fraud_transactions": int(y_true.sum()),
        "legitimate_transactions": int((y_true == 0).sum()),
        "fraud_prevalence": float(y_true.mean()),
        "strategy_count": len(EXPECTED_STRATEGIES),
        "strategies": EXPECTED_STRATEGIES,
        "metrics": [
            "average_precision",
            "roc_auc",
            "brier_score",
            "log_loss",
            "ece",
        ],
        "ece_definition": "Equal-width bins, 10 bins, weighted absolute calibration gap.",
        "prediction_sha256": actual_prediction_hash,
        "prediction_metadata_sha256": actual_metadata_hash,
        "prediction_file": str(PREDICTION_PATH.relative_to(ROOT)),
        "metrics_file": str(METRICS_PATH.relative_to(ROOT)),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nFINAL HOLDOUT EVALUATION COMPLETED")
    print("=" * 100)
    print(results_df.to_string(index=False))
    print(f"\nMetrics: {METRICS_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
