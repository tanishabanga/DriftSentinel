
"""Evaluate how review capacity affects fraud-screening outcomes."""

import numpy as np
import pandas as pd

from driftsentinel.policies.capacity import (
    make_capacity_constrained_decisions,
)

CAPACITIES = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 895]


def evaluate_capacity(df, capacity):
    decisions = make_capacity_constrained_decisions(
        probabilities=df["probability"],
        review_capacity=capacity,
    )

    fraud = df["isFraud"].to_numpy() == 1
    legitimate = ~fraud

    actions = decisions["action"].to_numpy()

    fraud_total = fraud.sum()
    legitimate_total = legitimate.sum()
    review_count = np.sum(actions == "review")

    fraud_blocked = np.sum(fraud & (actions == "block"))
    fraud_reviewed = np.sum(fraud & (actions == "review"))
    fraud_approved = np.sum(fraud & (actions == "approve"))

    legitimate_blocked = np.sum(
        legitimate & (actions == "block")
    )

    return {
        "capacity": capacity,
        "actual_review_count": review_count,
        "review_utilization": (
            review_count / capacity if capacity > 0 else 0.0
        ),
        "total_expected_cost": decisions["expected_cost"].sum(),
        "fraud_intervention_rate": (
            (fraud_blocked + fraud_reviewed) / fraud_total
        ),
        "fraud_approval_rate": fraud_approved / fraud_total,
        "legitimate_block_rate": (
            legitimate_blocked / legitimate_total
        ),
        "fraud_approved": fraud_approved,
        "fraud_reviewed": fraud_reviewed,
        "fraud_blocked": fraud_blocked,
        "legitimate_blocked": legitimate_blocked,
    }


def main():
    input_path = (
        "experiments/results/"
        "logistic_regression_pilot_predictions.parquet"
    )
    output_path = (
        "experiments/results/capacity_sensitivity.csv"
    )

    print("Loading pilot predictions...")
    df = pd.read_parquet(input_path)

    results = [
        evaluate_capacity(df, capacity)
        for capacity in CAPACITIES
    ]

    comparison = pd.DataFrame(results)

    # Marginal cost reduction relative to the previous capacity.
    comparison["cost_reduction"] = (
        -comparison["total_expected_cost"].diff()
    )

    comparison["additional_review_slots"] = (
        comparison["capacity"].diff()
    )

    comparison["cost_reduction_per_extra_slot"] = (
        comparison["cost_reduction"]
        / comparison["additional_review_slots"]
    )

    comparison.to_csv(output_path, index=False)

    print("\nCAPACITY SENSITIVITY RESULTS")
    print(comparison.round(4).to_string(index=False))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
