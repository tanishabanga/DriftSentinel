
import json
import logging
from pathlib import Path

import pandas as pd

from driftsentinel.data.splitting import chronological_split

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/processed/ieee_cis_train.parquet"
OUTPUT_DIR = ROOT / "data/processed/splits"
METADATA_FILE = ROOT / "experiments/results/split_metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def create_splits() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading integrated Parquet dataset...")
    df = pd.read_parquet(INPUT_FILE)

    logger.info("Creating chronological partitions...")

    result = chronological_split(
        df,
        timestamp_column="TransactionDT",
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    partitions = {
        "train": result.train,
        "validation": result.validation,
        "test": result.test,
    }

    metadata = {
        "strategy": "chronological",
        "configured_ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "partitions": {},
    }

    for name, partition in partitions.items():
        logger.info("Saving %s partition...", name)

        output_path = OUTPUT_DIR / f"{name}.parquet"

        partition.to_parquet(
            output_path,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )

        metadata["partitions"][name] = {
            "rows": len(partition),
            "fraud_counts": {
                str(int(k)): int(v)
                for k, v in partition["isFraud"].value_counts().items()
            },
            "fraud_percentage": round(
                float(partition["isFraud"].mean() * 100), 4
            ),
            "timestamp_min": int(partition["TransactionDT"].min()),
            "timestamp_max": int(partition["TransactionDT"].max()),
        }

    # Verify that temporal partitions do not overlap.
    train_end = result.train["TransactionDT"].max()
    validation_start = result.validation["TransactionDT"].min()

    validation_end = result.validation["TransactionDT"].max()
    test_start = result.test["TransactionDT"].min()

    if not train_end < validation_start:
        raise ValueError("Train and validation timestamps overlap.")

    if not validation_end < test_start:
        raise ValueError("Validation and test timestamps overlap.")

    metadata["temporal_boundaries_valid"] = True

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n========== SPLIT SUMMARY ==========")

    for name, details in metadata["partitions"].items():
        print(f"\n{name.upper()}")
        print("Rows:", details["rows"])
        print("Fraud counts:", details["fraud_counts"])
        print("Fraud percentage:", details["fraud_percentage"])
        print(
            "Timestamp:",
            details["timestamp_min"],
            "to",
            details["timestamp_max"],
        )

    print("\nTemporal boundaries valid:", metadata["temporal_boundaries_valid"])
    print("Saved partitions:", OUTPUT_DIR)
    print("Saved metadata:", METADATA_FILE)


if __name__ == "__main__":
    create_splits()
