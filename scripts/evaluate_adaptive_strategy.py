
"""Evaluate static and adaptive calibration strategies under review capacity."""

from pathlib import Path

import pandas as pd

from driftsentinel.calibration.calibrator import ProbabilityCalibrator
from driftsentinel.policies.capacity import (
    make_capacity_constrained_decisions,
)
from driftsentinel.policies.costs import DecisionCosts

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    ROOT / "experiments" / "results" / "temporal_predictions"
)
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
        raise ValueError(f"Missing columns in {filename}: {missing}")

    return df


def calculate_metrics(
    strategy: str,
    y_true: pd.Series,
    probabilities,
    costs: DecisionCosts,
) -> dict:
    decisions = make_capacity_constrained_decisions(
        probabilities=probabilities,
        review_capacity=REVIEW_CAPACITY,
        costs=costs,
    )

    actions = decisions["action"].to_numpy()
    y = y_true.to_numpy()

    fraud = y == 1
    legitimate = y == 0

    approved = actions == "approve"
    reviewed = actions == "review"
    blocked = actions == "block"

    fraud_count = int(fraud.sum())
    legitimate_count = int(legitimate.sum())

    fraud_approved = int((approved & fraud).sum())
    fraud_reviewed = int((reviewed & fraud).sum())
    fraud_blocked = int((blocked & fraud).sum())

    legitimate_approved = int((approved & legitimate).sum())
    legitimate_reviewed = int((reviewed & legitimate).sum())
    legitimate_blocked = int((blocked & legitimate).sum())

    review_count = int(reviewed.sum())

    realized_cost = (
        fraud_approved * costs.approve_fraud
        + fraud_reviewed * costs.review_fraud
        + fraud_blocked * costs.block_fraud
        + legitimate_approved * costs.approve_legitimate
        + legitimate_reviewed * costs.review_legitimate
        + legitimate_blocked * costs.block_legitimate
    )

    return {
        "strategy": strategy,
        "model_estimated_expected_cost": float(
            decisions["expected_cost"].sum()
        ),
        "realized_cost": float(realized_cost),
        "review_capacity": REVIEW_CAPACITY,
        "actual_reviews": review_count,
        "capacity_utilization": review_count / REVIEW_CAPACITY,
        "fraud_approved": fraud_approved,
        "fraud_reviewed": fraud_reviewed,
        "fraud_blocked": fraud_blocked,
        "legitimate_approved": legitimate_approved,
        "legitimate_reviewed": legitimate_reviewed,
        "legitimate_blocked": legitimate_blocked,
        "fraud_intervention_rate": (
            (fraud_reviewed + fraud_blocked) / fraud_count
            if fraud_count else 0.0
        ),
        "fraud_approval_rate": (
            fraud_approved / fraud_count if fraud_count else 0.0
        ),
        "legitimate_block_rate": (
            legitimate_blocked / legitimate_count
            if legitimate_count else 0.0
        ),
    }


def main() -> None:
    calibration_df = load_predictions(
        "initial_calibration_predictions.parquet"
    )
    adaptation_df = load_predictions(
        "adaptation_predictions.parquet"
    )
    evaluation_df = load_predictions(
        "evaluation_predictions.parquet"
    )

    y_calibration = calibration_df["isFraud"]
    p_calibration = calibration_df["probability"]

    y_adaptation = adaptation_df["isFraud"]
    p_adaptation = adaptation_df["probability"]

    y_evaluation = evaluation_df["isFraud"]
    p_evaluation = evaluation_df["probability"]

    # Fit static calibrators on the initial calibration window.
    static_sigmoid = ProbabilityCalibrator(method="sigmoid")
    static_sigmoid.fit(y_calibration, p_calibration)

    static_isotonic = ProbabilityCalibrator(method="isotonic")
    static_isotonic.fit(y_calibration, p_calibration)

    # Fit adaptive calibrators on the newer adaptation window.
    adaptive_sigmoid = ProbabilityCalibrator(method="sigmoid")
    adaptive_sigmoid.fit(y_adaptation, p_adaptation)

    adaptive_isotonic = ProbabilityCalibrator(method="isotonic")
    adaptive_isotonic.fit(y_adaptation, p_adaptation)

    strategies = {
        "uncalibrated": p_evaluation,
        "static_sigmoid": static_sigmoid.predict(p_evaluation),
        "static_isotonic": static_isotonic.predict(p_evaluation),
        "adaptive_sigmoid": adaptive_sigmoid.predict(p_evaluation),
        "adaptive_isotonic": adaptive_isotonic.predict(p_evaluation),
    }

    costs = DecisionCosts()
    results = []

    for name, probabilities in strategies.items():
        results.append(
            calculate_metrics(
                strategy=name,
                y_true=y_evaluation,
                probabilities=probabilities,
                costs=costs,
            )
        )

    results_df = pd.DataFrame(results)

    output_path = RESULT_DIR / "adaptive_strategy_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\nADAPTIVE DECISION STRATEGY COMPARISON")
    print("=" * 110)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
