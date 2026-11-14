
"""Profile temporal transaction density for block bootstrap design."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PATH = (
    ROOT
    / "experiments/results/temporal_predictions/"
    / "evaluation_frozen_predictions.parquet"
)


def main() -> None:
    df = pd.read_parquet(PATH)

    # TransactionDT represents elapsed time in seconds.
    # Group observations into UTC-like 24-hour elapsed-time bins.
    origin = int(df["TransactionDT"].min())

    df["elapsed_day"] = (
        (df["TransactionDT"] - origin) // 86_400
    ).astype(int)

    daily = (
        df.groupby("elapsed_day")
        .agg(
            transactions=("TransactionID", "size"),
            fraud_cases=("isFraud", "sum"),
            first_timestamp=("TransactionDT", "min"),
            last_timestamp=("TransactionDT", "max"),
        )
        .reset_index()
    )

    daily["fraud_rate"] = (
        daily["fraud_cases"] / daily["transactions"]
    )

    print("EVALUATION TEMPORAL PROFILE")
    print("=" * 80)
    print(daily.to_string(index=False))

    print("\nSUMMARY")
    print(f"Total transactions: {len(df):,}")
    print(f"Number of elapsed days: {len(daily)}")
    print(f"Smallest daily count: {daily['transactions'].min():,}")
    print(f"Largest daily count: {daily['transactions'].max():,}")
    print(f"Average daily count: {daily['transactions'].mean():,.1f}")


if __name__ == "__main__":
    main()
