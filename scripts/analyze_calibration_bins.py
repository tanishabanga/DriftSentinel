"""Analyze bin-level calibration errors across temporal strategies."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = ROOT / "experiments" / "results"

INPUT_PATH = RESULT_DIR / "temporal_calibration_bins.csv"

DETAIL_OUTPUT = RESULT_DIR / "calibration_bin_diagnostics.csv"

SUMMARY_OUTPUT = RESULT_DIR / "calibration_strategy_diagnostics.csv"

def analyze_calibration_bins() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate bin-level calibration gaps and strategy summaries."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Calibration bin file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "bin",
        "transaction_count",
        "mean_probability",
        "observed_fraud_rate",
        "strategy",
        "group",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Calibration bin file is empty.")

    if (df["transaction_count"] <= 0).any():
        raise ValueError("Transaction counts must be positive.")

    probability_columns = [
        "mean_probability",
        "observed_fraud_rate",
    ]

    if not np.isfinite(df[probability_columns].to_numpy()).all():
        raise ValueError("Calibration values must be finite.")

    if not df[probability_columns].apply(
        lambda column: column.between(0, 1).all()
    ).all():
        raise ValueError("Calibration values must be between 0 and 1.")

    # Signed gap: positive means underprediction of fraud frequency.
    df["signed_calibration_gap"] = (
        df["observed_fraud_rate"] - df["mean_probability"]
    )

    df["absolute_calibration_gap"] = (
        df["signed_calibration_gap"].abs()
    )

    # Each bin's weighted contribution to quantile-bin ECE.
    df["strategy_total"] = df.groupby(
        "strategy"
    )["transaction_count"].transform("sum")

    df["weighted_gap_contribution"] = (
        df["transaction_count"] / df["strategy_total"]
    ) * df["absolute_calibration_gap"]

    # Summarize the calibration error for each strategy.
    summary = (
        df.groupby(["group", "strategy"], as_index=False)
        .agg(
            total_transactions=("transaction_count", "sum"),
            number_of_bins=("bin", "count"),
            weighted_calibration_error=(
                "weighted_gap_contribution",
                "sum",
            ),
            mean_absolute_bin_gap=(
                "absolute_calibration_gap",
                "mean",
            ),
            maximum_absolute_bin_gap=(
                "absolute_calibration_gap",
                "max",
            ),
        )
    )

    summary = summary.sort_values(
        "weighted_calibration_error"
    ).reset_index(drop=True)

    return df, summary

def main() -> None:
    """Generate calibration-bin diagnostic reports."""

    detail, summary = analyze_calibration_bins()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    detail.to_csv(DETAIL_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("\nCALIBRATION BIN DIAGNOSTICS")
    print("=" * 90)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(f"\nDetailed report: {DETAIL_OUTPUT}")
    print(f"Strategy summary: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()