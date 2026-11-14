
"""Compare static fraud decision strategies under a fixed review capacity."""

from pathlib import Path

import pandas as pd

from driftsentinel.calibration.calibrator import ProbabilityCalibrator
from driftsentinel.policies.capacity import (
    make_capacity_constrained_decisions,
)
from driftsentinel.policies.costs import DecisionCosts

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = ROOT / "experiments" / "results" / "temporal_predictions"
RESULT_DIR = ROOT / "experiments" / "results"

REVIEW_CAPACITY = 200


def load_predictions(filename: str) -> pd.DataFrame:
    path = PREDICTION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_parquet(path)

    required = {"isFraud", "probability"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def calculate_strategy_metrics(
    name: str,
    y_true: pd.Series,
    probabilities,
) -> dict:
    decisions = make_capacity_constrained_decisions(
        probabilities=probabilities,
        review_capacity=REVIEW_CAPACITY,
    )

    actions = decisions["action"].to_numpy()
    y = y_true.to_numpy()

    fraud_mask = y == 1
    legitimate_mask = y == 0

    fraud_count = int(fraud_mask.sum())
    legitimate_count = int(legitimate_mask.sum())

    approved_fraud = int(
        ((actions == "approve") & fraud_mask).sum()
    )
    reviewed_fraud = int(
        ((actions == "review") & fraud_mask).sum()
    )
    blocked_fraud = int(
        ((actions == "block") & fraud_mask).sum()
    )
    blocked_legitimate = int(
        ((actions == "block") & legitimate_mask).sum()
    )

    review_count = int((actions == "review").sum())

    costs = DecisionCosts()

    # Calculate cost using actual evaluation labels and chosen actions.
    realized_cost = (
        approved_fraud * costs.approve_fraud
        + reviewed_fraud * costs.review_fraud
        + blocked_fraud * costs.block_fraud
        + blocked_legitimate * costs.block_legitimate
        + int(((actions == "approve") & legitimate_mask).sum())
        * costs.approve_legitimate
        + int(((actions == "review") & legitimate_mask).sum())
        * costs.review_legitimate
    )

    return {
        "strategy": name,
        "model_estimated_expected_cost": float(
            decisions["expected_cost"].sum()
        ),
        "realized_cost": float(realized_cost),
        "review_capacity": REVIEW_CAPACITY,
        "actual_reviews": review_count,
        "capacity_utilization": review_count / REVIEW_CAPACITY,
        "fraud_approved": approved_fraud,
        "fraud_reviewed": reviewed_fraud,
        "fraud_blocked": blocked_fraud,
        "legitimate_blocked": blocked_legitimate,
        "fraud_intervention_rate": (
            (reviewed_fraud + blocked_fraud) / fraud_count
            if fraud_count else 0.0
        ),
        "fraud_approval_rate": (
            approved_fraud / fraud_count if fraud_count else 0.0
        ),
        "legitimate_block_rate": (
            blocked_legitimate / legitimate_count
            if legitimate_count else 0.0
        ),
    }


def main() -> None:
    calibration_df = load_predictions(
        "initial_calibration_predictions.parquet"
    )
    evaluation_df = load_predictions(
        "evaluation_predictions.parquet"
    )

    y_calibration = calibration_df["isFraud"]
    p_calibration = calibration_df["probability"]

    y_evaluation = evaluation_df["isFraud"]
    p_uncalibrated = evaluation_df["probability"]

    # Fit calibrators ONLY on the initial calibration window.
    sigmoid = ProbabilityCalibrator(method="sigmoid")
    sigmoid.fit(y_calibration, p_calibration)

    isotonic = ProbabilityCalibrator(method="isotonic")
    isotonic.fit(y_calibration, p_calibration)

    # Transform probabilities in the later evaluation window.
    p_sigmoid = sigmoid.predict(p_uncalibrated)
    p_isotonic = isotonic.predict(p_uncalibrated)

    strategies = [
        ("uncalibrated", p_uncalibrated),
        ("sigmoid", p_sigmoid),
        ("isotonic", p_isotonic),
    ]

    results = []

    for name, probabilities in strategies:
        metrics = calculate_strategy_metrics(
            name=name,
            y_true=y_evaluation,
            probabilities=probabilities,
        )
        results.append(metrics)

    results_df = pd.DataFrame(results)

    output_path = RESULT_DIR / "static_strategy_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\nSTATIC STRATEGY COMPARISON")
    print("=" * 75)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
