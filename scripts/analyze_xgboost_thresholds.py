
"""Analyze XGBoost validation predictions across thresholds."""

from pathlib import Path

import pandas as pd

from driftsentinel.policies.threshold_analysis import analyze_thresholds

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT / "experiments/results/xgboost_full_predictions.parquet"
)

OUTPUT_PATH = (
    ROOT / "experiments/results/xgboost_threshold_analysis.csv"
)


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    predictions_df = pd.read_parquet(PREDICTIONS_PATH)

    required_columns = {"isFraud", "probability"}

    if not required_columns.issubset(predictions_df.columns):
        raise ValueError(
            f"Predictions must contain: {required_columns}"
        )

    y_true = predictions_df["isFraud"]
    probabilities = predictions_df["probability"]

    thresholds = [round(value, 2) for value in
                  [0.05 + i * 0.05 for i in range(19)]]

    print("XGBOOST THRESHOLD ANALYSIS")
    print(f"Validation rows: {len(predictions_df):,}")
    print(f"Fraud prevalence: {y_true.mean():.4%}")
    print(f"Thresholds tested: {len(thresholds)}")

    results = analyze_thresholds(
        y_true=y_true,
        probabilities=probabilities,
        thresholds=thresholds,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nTHRESHOLD COMPARISON")
    print(
        results[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "true_positives",
                "false_positives",
                "false_negatives",
                "flagged_count",
                "flagged_rate",
            ]
        ].round(4).to_string(index=False)
    )

    best_f1 = results.loc[results["f1"].idxmax()]
    best_recall = results.loc[results["recall"].idxmax()]

    print("\nBEST F1 THRESHOLD")
    print(best_f1.to_string())

    print("\nBEST RECALL THRESHOLD")
    print(best_recall.to_string())

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
