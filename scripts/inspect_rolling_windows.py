
"""Inspect candidate chronological windows for rolling evaluation."""

from pathlib import Path

import pandas as pd

from driftsentinel.experiments.temporal import (
    split_into_temporal_segments,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/processed/splits/train.parquet"
VALIDATION_PATH = ROOT / "data/processed/splits/validation.parquet"

TIMESTAMP = "TransactionDT"
TARGET = "isFraud"


def describe(name: str, df: pd.DataFrame) -> dict:
    return {
        "partition": name,
        "rows": len(df),
        "fraud_count": int(df[TARGET].sum()),
        "fraud_rate_pct": round(df[TARGET].mean() * 100, 4),
        "timestamp_min": int(df[TIMESTAMP].min()),
        "timestamp_max": int(df[TIMESTAMP].max()),
    }


def main() -> None:
    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[TIMESTAMP, TARGET],
    )
    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=[TIMESTAMP, TARGET],
    )

    train = train.sort_values(TIMESTAMP, kind="mergesort")
    validation = validation.sort_values(TIMESTAMP, kind="mergesort")

    # Candidate historical allocation:
    # 80% initial model training, 20% initial calibration.
    train_boundary = train[TIMESTAMP].quantile(0.80)

    initial_train = train[
        train[TIMESTAMP] <= train_boundary
    ]
    initial_calibration = train[
        train[TIMESTAMP] > train_boundary
    ]


    validation_segments = split_into_temporal_segments(
        validation,
        n_segments=5,
        timestamp_column=TIMESTAMP,
    )

    ordered_windows = [
        (f"validation_segment_{index + 1}", window)
        for index, window in enumerate(validation_segments)
    ]

    partitions = [
        describe("initial_train", initial_train),
        describe("initial_calibration", initial_calibration),
        *[
            describe(name, window)
            for name, window in ordered_windows
        ],
    ]

    print("\nCANDIDATE ROLLING WINDOW PROFILE")
    print(pd.DataFrame(partitions).to_string(index=False))

    print("\nCandidate validation segment sizes:")
    print([len(window) for _, window in ordered_windows])

    print("\nThis is an inspection only. No data was modified.")


if __name__ == "__main__":
    main()
