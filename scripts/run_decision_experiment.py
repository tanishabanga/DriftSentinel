
"""Run the cost-sensitive decision engine on pilot predictions."""

import pandas as pd

from driftsentinel.policies.decision import make_decisions


def main():
    input_path = (
        "experiments/results/logistic_regression_pilot_predictions.parquet"
    )

    output_path = "experiments/results/decision_report.csv"

    print("Loading validation predictions...")
    df = pd.read_parquet(input_path)

    decisions = make_decisions(df["probability"])

    report = df[
        ["TransactionID", "TransactionDT", "isFraud"]
    ].copy()

    report["probability"] = df["probability"]
    report["action"] = decisions["action"]
    report["expected_cost"] = decisions["expected_cost"]

    report.to_csv(output_path, index=False)

    print("\nDECISION DISTRIBUTION")
    print(report["action"].value_counts())

    print("\nDECISION DISTRIBUTION BY ACTUAL CLASS")
    print(pd.crosstab(report["isFraud"], report["action"]))

    print("\nTOTAL EXPECTED COST")
    print(round(report["expected_cost"].sum(), 2))

    print(f"\nDecision report saved to: {output_path}")


if __name__ == "__main__":
    main()
