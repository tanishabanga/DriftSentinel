
"""Profile transaction density in six-hour temporal blocks."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PATH = (
    ROOT
    / "experiments/results/temporal_predictions/"
    / "evaluation_frozen_predictions.parquet"
)

BLOCK_SECONDS = 6 * 60 * 60


def main() -> None:
    df = pd.read_parquet(PATH)

    origin = int(df["TransactionDT"].min())

    df["elapsed_block"] = (
        (df["TransactionDT"] - origin) // BLOCK_SECONDS
    ).astype(int)

    blocks = (
        df.groupby("elapsed_block")
        .agg(
            transactions=("TransactionID", "size"),
            fraud_cases=("isFraud", "sum"),
            first_timestamp=("TransactionDT", "min"),
            last_timestamp=("TransactionDT", "max"),
        )
        .reset_index()
    )

    blocks["fraud_rate"] = (
        blocks["fraud_cases"] / blocks["transactions"]
    )

    print("SIX-HOUR TEMPORAL BLOCK PROFILE")
    print("=" * 90)
    print(blocks.to_string(index=False))

    print("\nSUMMARY")
    print(f"Total transactions: {len(df):,}")
    print(f"Number of blocks: {len(blocks)}")
    print(f"Smallest block: {blocks['transactions'].min():,}")
    print(f"Largest block: {blocks['transactions'].max():,}")
    print(f"Average block size: {blocks['transactions'].mean():,.1f}")
    print(f"Total fraud cases: {int(df['isFraud'].sum()):,}")


if __name__ == "__main__":
    main()
