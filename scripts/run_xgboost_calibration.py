
"""Evaluate chronological probability calibration for XGBoost."""

import json
from pathlib import Path

import joblib
import pandas as pd

from driftsentinel.calibration.calibrator import ProbabilityCalibrator
from driftsentinel.calibration.metrics import evaluate_calibration

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT / "experiments/results/xgboost_full_predictions.parquet"
)

RESULTS_DIR = ROOT / "experiments/results"
ARTIFACT_DIR = ROOT / "artifacts/xgboost/calibration"

COMPARISON_PATH = RESULTS_DIR / "xgboost_calibration_comparison.csv"
METADATA_PATH = RESULTS_DIR / "xgboost_calibration_metadata.json"
EVALUATION_PATH = (
    RESULTS_DIR / "xgboost_calibrated_evaluation_predictions.parquet"
)


def load_predictions() -> pd.DataFrame:
    """Load and validate XGBoost validation predictions."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_parquet(PREDICTIONS_PATH)

    required = {
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "probability",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Prediction dataset is empty.")

    if df["TransactionID"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found.")

    df = df.sort_values(
        "TransactionDT",
        kind="stable",
    ).reset_index(drop=True)

    return df


def validate_segment(df: pd.DataFrame, name: str) -> None:
    """Ensure a chronological segment contains both classes."""

    if df.empty:
        raise ValueError(f"{name} segment is empty.")

    if not df["isFraud"].isin([0, 1]).all():
        raise ValueError(f"Invalid labels in {name} segment.")

    if df["isFraud"].nunique() < 2:
        raise ValueError(
            f"{name} segment must contain both classes."
        )


def main() -> None:
    print("Loading XGBoost validation predictions...")

    df = load_predictions()

    probabilities = df["probability"].to_numpy(dtype=float)

    if not pd.Series(probabilities).between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not pd.Series(probabilities).map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all():
        raise ValueError("Probabilities must be finite.")

    # Chronological 50/50 calibration and evaluation split.
    split_index = len(df) // 2

    calibration_df = df.iloc[:split_index].copy()
    evaluation_df = df.iloc[split_index:].copy()

    validate_segment(calibration_df, "Calibration")
    validate_segment(evaluation_df, "Evaluation")

    y_calibration = calibration_df["isFraud"].to_numpy()
    p_calibration = calibration_df["probability"].to_numpy(
        dtype=float
    )

    y_evaluation = evaluation_df["isFraud"].to_numpy()
    p_evaluation = evaluation_df["probability"].to_numpy(
        dtype=float
    )

    print(f"Total validation rows: {len(df):,}")
    print(f"Calibration rows: {len(calibration_df):,}")
    print(f"Evaluation rows: {len(evaluation_df):,}")
    print(f"Calibration fraud rate: {y_calibration.mean():.4%}")
    print(f"Evaluation fraud rate: {y_evaluation.mean():.4%}")

    # Evaluate original probabilities first.
    results = {
        "uncalibrated": evaluate_calibration(
            y_evaluation,
            p_evaluation,
        )
    }

    calibrated_predictions = {
        "TransactionID": evaluation_df["TransactionID"].to_numpy(),
        "TransactionDT": evaluation_df["TransactionDT"].to_numpy(),
        "isFraud": y_evaluation,
        "probability_uncalibrated": p_evaluation,
    }

    fitted_calibrators = {}

    for method in ("sigmoid", "isotonic"):
        print(f"\nFitting {method} calibrator...")

        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(y_calibration, p_calibration)

        calibrated = calibrator.predict(p_evaluation)

        results[method] = evaluate_calibration(
            y_evaluation,
            calibrated,
        )

        calibrated_predictions[
            f"probability_{method}"
        ] = calibrated

        fitted_calibrators[method] = calibrator

    comparison = pd.DataFrame(results).T
    comparison.index.name = "method"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(COMPARISON_PATH)

    pd.DataFrame(calibrated_predictions).to_parquet(
        EVALUATION_PATH,
        index=False,
    )

    # Save fitted calibrators for later inference.
    for method, calibrator in fitted_calibrators.items():
        joblib.dump(
            calibrator,
            ARTIFACT_DIR / f"{method}_calibrator.joblib",
        )

    metadata = {
        "experiment": "xgboost_chronological_calibration",
        "status": "completed",
        "base_model": "XGBoost",
        "calibration_method_comparison": [
            "uncalibrated",
            "sigmoid",
            "isotonic",
        ],
        "split_method": "chronological_50_50",
        "total_validation_rows": len(df),
        "calibration_rows": len(calibration_df),
        "evaluation_rows": len(evaluation_df),
        "calibration_fraud_rate": float(y_calibration.mean()),
        "evaluation_fraud_rate": float(y_evaluation.mean()),
        "calibration_start_timestamp": int(
            calibration_df["TransactionDT"].iloc[0]
        ),
        "calibration_end_timestamp": int(
            calibration_df["TransactionDT"].iloc[-1]
        ),
        "evaluation_start_timestamp": int(
            evaluation_df["TransactionDT"].iloc[0]
        ),
        "evaluation_end_timestamp": int(
            evaluation_df["TransactionDT"].iloc[-1]
        ),
        "test_set_used": False,
        "metrics": results,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nCALIBRATION COMPARISON")
    print(comparison.round(6).to_string())

    print("\nARTIFACTS SAVED")
    print(COMPARISON_PATH)
    print(METADATA_PATH)
    print(EVALUATION_PATH)
    print(ARTIFACT_DIR)


if __name__ == "__main__":
    main()
