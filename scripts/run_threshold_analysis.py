
"""Run threshold analysis on saved Logistic Regression predictions."""

import pandas as pd

from driftsentinel.policies.threshold_analysis import analyze_thresholds


def main():
    predictions_path = (
        "experiments/results/logistic_regression_pilot_predictions.parquet"
    )

    results_path = "experiments/results/threshold_analysis.csv"

    print("Loading saved validation predictions...")

    predictions_df = pd.read_parquet(predictions_path)

    y_true = predictions_df["isFraud"]
    probabilities = predictions_df["probability"]

    print(f"Validation rows: {len(predictions_df):,}")
    print(f"Fraud prevalence: {y_true.mean():.4%}")

    print("\nAnalyzing classification thresholds...")

    results = analyze_thresholds(
        y_true=y_true,
        probabilities=probabilities,
    )

    results.to_csv(results_path, index=False)

    print("\nTHRESHOLD COMPARISON")
    print(
        results[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_positives",
                "false_negatives",
                "flagged_count",
                "flagged_rate",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
