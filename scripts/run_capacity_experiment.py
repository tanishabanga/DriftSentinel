
"""Compare fraud decisions under different review capacities."""

import pandas as pd

from driftsentinel.policies.capacity import (
    make_capacity_constrained_decisions,
)
from driftsentinel.policies.decision import make_decisions


def evaluate_policy(df, decisions, policy_name):
    report = df[["TransactionID", "TransactionDT", "isFraud"]].copy()
    report["action"] = decisions["action"]
    report["expected_cost"] = decisions["expected_cost"]

    fraud = report["isFraud"] == 1
    legitimate = report["isFraud"] == 0

    return {
        "policy": policy_name,
        "total_expected_cost": decisions["expected_cost"].sum(),
        "review_count": (decisions["action"] == "review").sum(),
        "review_rate": (decisions["action"] == "review").mean(),
        "fraud_approved": (
            fraud & (report["action"] == "approve")
        ).sum(),
        "fraud_reviewed": (
            fraud & (report["action"] == "review")
        ).sum(),
        "fraud_blocked": (
            fraud & (report["action"] == "block")
        ).sum(),
        "legitimate_blocked": (
            legitimate & (report["action"] == "block")
        ).sum(),
    }


def main():
    input_path = (
        "experiments/results/"
        "logistic_regression_pilot_predictions.parquet"
    )

    print("Loading pilot predictions...")
    df = pd.read_parquet(input_path)

    results = []

    # Unconstrained baseline
    baseline = make_decisions(df["probability"])
    results.append(
        evaluate_policy(df, baseline, "unconstrained")
    )

    # Capacity-constrained experiments
    for capacity in [100, 200, 500]:
        decisions = make_capacity_constrained_decisions(
            df["probability"],
            review_capacity=capacity,
        )

        results.append(
            evaluate_policy(
                df,
                decisions,
                f"capacity_{capacity}",
            )
        )

    comparison = pd.DataFrame(results)

    output_path = (
        "experiments/results/capacity_comparison.csv"
    )

    comparison.to_csv(output_path, index=False)

    print("\nCAPACITY EXPERIMENT RESULTS")
    print(comparison.to_string(index=False))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
