
"""Prepare chronological datasets for temporal adaptation experiments."""

import json
from pathlib import Path

import pandas as pd

from driftsentinel.data.sampling import read_parquet_sample
from driftsentinel.data.temporal import (
    split_adaptation_window,
    split_temporal_windows,
)

TRAIN_PATH = Path("data/processed/splits/train.parquet")
VALIDATION_PATH = Path("data/processed/splits/validation.parquet")
TEST_PATH = Path("data/processed/splits/test.parquet")

OUTPUT_DIR = Path("data/processed/temporal")
RESULTS_DIR = Path("experiments/results")

TRAIN_SAMPLE_SIZE = 20_000
RANDOM_STATE = 42


def describe_partition(name, df):
    return {
        "name": name,
        "rows": len(df),
        "fraud_count": int(df["isFraud"].sum()),
        "fraud_percentage": round(
            df["isFraud"].mean() * 100, 4
        ),
        "timestamp_min": int(df["TransactionDT"].min()),
        "timestamp_max": int(df["TransactionDT"].max()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading initial training sample...")

    train_sample = read_parquet_sample(
        TRAIN_PATH,
        n_rows=TRAIN_SAMPLE_SIZE,
    )

    train_sample = train_sample.sort_values(
        "TransactionDT"
    ).reset_index(drop=True)

    print("Splitting validation into temporal windows...")

    validation = pd.read_parquet(VALIDATION_PATH)

    windows = split_temporal_windows(validation)

    adaptation_split = split_adaptation_window(
        windows.adaptation
    )

    print("Checking final test boundary...")

    test = pd.read_parquet(
        TEST_PATH,
        columns=["TransactionDT", "TransactionID"],
    )

    assert (
        windows.evaluation["TransactionDT"].max()
        < test["TransactionDT"].min()
    )

    partitions = {
        "initial_train_sample": train_sample,
        "initial_calibration": windows.initial_calibration,
        "adaptation_train": adaptation_split.adaptation_train,
        "adaptation_calibration": (
            adaptation_split.adaptation_calibration
        ),
        "evaluation": windows.evaluation,
    }

    metadata = {
        "experiment": "temporal_retraining_pilot",
        "random_state": RANDOM_STATE,
        "training_sample_size": TRAIN_SAMPLE_SIZE,
        "final_test_used": False,
        "partitions": [],
    }

    for name, df in partitions.items():
        output_path = OUTPUT_DIR / f"{name}.parquet"

        df.to_parquet(output_path, index=False)

        metadata["partitions"].append(
            describe_partition(name, df)
        )

        print(f"Saved {name}: {len(df):,} rows")

    metadata_path = (
        RESULTS_DIR / "temporal_experiment_metadata.json"
    )

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nTEMPORAL EXPERIMENT DATA PREPARATION")
    print(
        pd.DataFrame(metadata["partitions"]).to_string(
            index=False
        )
    )

    print(
        "\nFinal test partition was not used "
        "for training or adaptation."
    )
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
