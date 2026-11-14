
"""Create a deterministic, temporally distributed 20k training sample."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INITIAL_PATH = ROOT / "data/processed/temporal/initial_train_sample.parquet"
ADAPTATION_PATH = ROOT / "data/processed/temporal/adaptation_train.parquet"

OUTPUT_PATH = ROOT / "data/processed/temporal/matched_size_train.parquet"
METADATA_PATH = ROOT / "experiments/results/matched_size_dataset_metadata.json"

TARGET = "isFraud"
TARGET_SIZE = 20_000


def main() -> None:
    initial = pd.read_parquet(INITIAL_PATH)
    adaptation = pd.read_parquet(ADAPTATION_PATH)

    combined = pd.concat(
        [initial, adaptation],
        ignore_index=True,
    )

    if combined["TransactionID"].duplicated().any():
        raise ValueError("Duplicate transaction IDs detected.")

    combined = combined.sort_values(
        ["TransactionDT", "TransactionID"],
        kind="stable",
    ).reset_index(drop=True)

    if len(combined) < TARGET_SIZE:
        raise ValueError("Not enough observations for matched-size sample.")

    # Select evenly spaced positions across the complete time range.
    positions = np.linspace(
        0,
        len(combined) - 1,
        num=TARGET_SIZE,
        dtype=int,
    )

    sample = combined.iloc[positions].copy()

    if len(sample) != TARGET_SIZE:
        raise ValueError("Incorrect matched sample size.")

    if sample["TransactionID"].duplicated().any():
        raise ValueError("Duplicate IDs in matched sample.")

    sample = sample.sort_values(
        ["TransactionDT", "TransactionID"],
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample.to_parquet(OUTPUT_PATH, index=False)

    metadata = {
        "experiment": "matched_size_temporal_training",
        "selection_method": "deterministic_evenly_spaced_positions",
        "source_rows": len(combined),
        "sample_rows": len(sample),
        "fraud_cases": int(sample[TARGET].sum()),
        "fraud_rate": float(sample[TARGET].mean()),
        "timestamp_start": int(sample["TransactionDT"].min()),
        "timestamp_end": int(sample["TransactionDT"].max()),
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("Matched-size dataset created")
    print(f"Source rows: {len(combined):,}")
    print(f"Selected rows: {len(sample):,}")
    print(f"Fraud cases: {int(sample[TARGET].sum()):,}")
    print(f"Fraud rate: {sample[TARGET].mean():.4%}")
    print(f"Timestamp range: {sample['TransactionDT'].min()} - "
          f"{sample['TransactionDT'].max()}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
