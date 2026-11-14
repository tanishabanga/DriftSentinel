
"""Compare frozen-model calibration with model retraining."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from driftsentinel.calibration.calibrator import ProbabilityCalibrator
from driftsentinel.calibration.metrics import evaluate_calibration

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    ROOT / "experiments" / "results" / "temporal_predictions"
)

RESULT_DIR = ROOT / "experiments" / "results"


def load_predictions(filename: str) -> pd.DataFrame:
    """Load and validate a prediction file."""

    path = PREDICTION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_parquet(path)

    required = {
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "probability",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {filename}: {missing}")

    if df["TransactionID"].duplicated().any():
        raise ValueError(f"Duplicate transaction IDs in {filename}.")

    if not df["probability"].between(0, 1).all():
        raise ValueError(f"Invalid probabilities in {filename}.")

    return df


def validate_same_transactions(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> None:
    """Ensure predictions refer to identical transactions and labels."""

    columns = ["TransactionID", "TransactionDT", "isFraud"]

    reference = (
        reference[columns]
        .sort_values("TransactionID")
        .reset_index(drop=True)
    )

    candidate = (
        candidate[columns]
        .sort_values("TransactionID")
        .reset_index(drop=True)
    )

    if not reference.equals(candidate):
        raise ValueError(
            "Prediction files contain different transaction "
            "IDs, timestamps, or labels."
        )

def evaluate_strategy(
    name: str,
    y_true: pd.Series,
    probabilities,
) -> dict:
    """Calculate probability and ranking metrics."""

    metrics = evaluate_calibration(y_true, probabilities)

    metrics["roc_auc"] = float(
        roc_auc_score(y_true, probabilities)
    )

    return {
        "strategy": name,
        **metrics,
    }


def main() -> None:
    initial_calibration = load_predictions(
        "initial_calibration_predictions.parquet"
    )

    adaptation_frozen = load_predictions(
        "adaptation_calibration_frozen_predictions.parquet"
    )

    adaptation_retrained = load_predictions(
        "adaptation_calibration_retrained_predictions.parquet"
    )

    evaluation_frozen = load_predictions(
        "evaluation_frozen_predictions.parquet"
    )

    evaluation_retrained = load_predictions(
        "evaluation_retrained_predictions.parquet"
    )

    adaptation_recent = load_predictions(
        "adaptation_calibration_recent_only_predictions.parquet"
    )

    evaluation_recent = load_predictions(
        "evaluation_recent_only_predictions.parquet"
    )

    adaptation_matched = load_predictions(
    "adaptation_calibration_matched_size_predictions.parquet"
)

    evaluation_matched = load_predictions(
        "evaluation_matched_size_predictions.parquet"
    )

    validate_same_transactions(
        adaptation_frozen,
        adaptation_retrained,
    )

    validate_same_transactions(
        evaluation_frozen,
        evaluation_retrained,
    )

    validate_same_transactions(
    adaptation_frozen,
    adaptation_recent,
    )

    validate_same_transactions(
        evaluation_frozen,
        evaluation_recent,
    )

    validate_same_transactions(
    adaptation_frozen,
    adaptation_matched,
    )

    validate_same_transactions(
        evaluation_frozen,
        evaluation_matched,
    )

    y_initial = initial_calibration["isFraud"]
    p_initial = initial_calibration["probability"]

    y_adaptation = adaptation_frozen["isFraud"]
    p_adaptation_frozen = adaptation_frozen["probability"]
    p_adaptation_retrained = adaptation_retrained["probability"]

    y_evaluation = evaluation_frozen["isFraud"]
    p_evaluation_frozen = evaluation_frozen["probability"]
    p_evaluation_retrained = evaluation_retrained["probability"]

    p_adaptation_recent = adaptation_recent["probability"]
    p_evaluation_recent = evaluation_recent["probability"]

    p_adaptation_matched = adaptation_matched["probability"]
    p_evaluation_matched = evaluation_matched["probability"]

    # Static calibration: original model + initial calibration window.
    static_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_initial,
        p_initial,
    )

    static_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_initial,
        p_initial,
    )

    # Adaptive calibration: frozen model + later calibration window.
    adaptive_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_adaptation,
        p_adaptation_frozen,
    )

    adaptive_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation,
        p_adaptation_frozen,
    )

    # Retrained model calibration: fit on its own predictions.
    retrained_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_adaptation,
        p_adaptation_retrained,
    )

    retrained_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation,
        p_adaptation_retrained,
    )

    recent_sigmoid = ProbabilityCalibrator("sigmoid").fit(
    y_adaptation,
    p_adaptation_recent,
    )

    recent_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation,
        p_adaptation_recent,
    )

    matched_sigmoid = ProbabilityCalibrator("sigmoid").fit(
    y_adaptation,
    p_adaptation_matched,
)

    matched_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation,
        p_adaptation_matched,
    )

    strategies = {
        "frozen_uncalibrated": p_evaluation_frozen,
        "frozen_static_sigmoid": static_sigmoid.predict(
            p_evaluation_frozen
        ),
        "frozen_static_isotonic": static_isotonic.predict(
            p_evaluation_frozen
        ),
        "frozen_adaptive_sigmoid": adaptive_sigmoid.predict(
            p_evaluation_frozen
        ),
        "frozen_adaptive_isotonic": adaptive_isotonic.predict(
            p_evaluation_frozen
        ),
        "retrained_uncalibrated": p_evaluation_retrained,
        "retrained_sigmoid": retrained_sigmoid.predict(
            p_evaluation_retrained
        ),
        "retrained_isotonic": retrained_isotonic.predict(
            p_evaluation_retrained
        ),
        "recent_only_uncalibrated": p_evaluation_recent,
        "recent_only_sigmoid": recent_sigmoid.predict(
            p_evaluation_recent
        ),
        "recent_only_isotonic": recent_isotonic.predict(
            p_evaluation_recent
        ),
        "matched_size_uncalibrated": p_evaluation_matched,

        "matched_size_sigmoid": matched_sigmoid.predict(
            p_evaluation_matched
        ),

        "matched_size_isotonic": matched_isotonic.predict(
            p_evaluation_matched
        ),
    }

    results = [
        evaluate_strategy(name, y_evaluation, probabilities)
        for name, probabilities in strategies.items()
    ]

    results_df = pd.DataFrame(results)

    output_path = RESULT_DIR / "retraining_calibration_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)

    print("\nRETRAINING AND CALIBRATION COMPARISON")
    print("=" * 110)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
