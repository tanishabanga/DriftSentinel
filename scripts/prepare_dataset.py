
import json
import logging
from pathlib import Path

from driftsentinel.data.loader import (
    load_identity,
    load_transactions,
)
from driftsentinel.data.validation import (
    validate_identity_data,
    validate_transaction_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"

OUTPUT_FILE = PROCESSED_DIR / "ieee_cis_train.parquet"
METADATA_FILE = RESULTS_DIR / "dataset_profile.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def prepare_dataset() -> None:
    """Prepare, validate and persist the integrated IEEE-CIS dataset."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading transaction dataset...")
    transactions = load_transactions(TRANSACTION_FILE)

    logger.info("Loading identity dataset...")
    identity = load_identity(IDENTITY_FILE)

    logger.info("Validating transaction data...")
    validate_transaction_data(transactions)

    logger.info("Validating identity data...")
    validate_identity_data(identity)

    # Confirm that identity features do not collide with
    # transaction features, apart from the join key.
    overlapping_columns = (
        set(transactions.columns)
        & set(identity.columns)
    ) - {"TransactionID"}

    if overlapping_columns:
        raise ValueError(
            f"Unexpected overlapping columns: {sorted(overlapping_columns)}"
        )

    logger.info("Merging transaction and identity data...")

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator="_identity_merge",
        sort=False,
    )

    # Measure identity coverage before removing the merge indicator.
    identity_matched = int(
        (merged["_identity_merge"] == "both").sum()
    )

    identity_unmatched = int(
        (merged["_identity_merge"] == "left_only").sum()
    )

    identity_coverage = identity_matched / len(merged) * 100

    merged.drop(columns="_identity_merge")

    if len(merged) != len(transactions):
        raise ValueError(
            "Merged row count differs from transaction row count."
        )

    logger.info("Calculating dataset statistics...")

    fraud_counts = (
        merged["isFraud"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    fraud_counts = {
        str(int(label)): int(count)
        for label, count in fraud_counts.items()
    }

    missingness = (
        merged.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    missingness_report = {
        column: round(float(value), 4)
        for column, value in missingness.items()
    }

    metadata = {
        "dataset": "IEEE-CIS Fraud Detection",
        "source_files": [
            "train_transaction.csv",
            "train_identity.csv",
        ],
        "rows": len(merged),
        "columns": len(merged.columns),
        "fraud_counts": fraud_counts,
        "fraud_percentage": round(
            float(merged["isFraud"].mean() * 100), 4
        ),
        "identity_rows": len(identity),
        "identity_matched_transactions": identity_matched,
        "identity_unmatched_transactions": identity_unmatched,
        "identity_coverage_percentage": round(
            identity_coverage, 4
        ),
        "timestamp_min": int(merged["TransactionDT"].min()),
        "timestamp_max": int(merged["TransactionDT"].max()),
        "missingness_percentage": missingness_report,
        "storage_format": "parquet",
    }

    logger.info("Saving processed dataset to Parquet...")

    merged.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    logger.info("Saving dataset metadata...")

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    logger.info("Dataset preparation completed.")

    print("\n========== DATASET SUMMARY ==========")
    print("Rows:", metadata["rows"])
    print("Columns:", metadata["columns"])
    print("Fraud percentage:", metadata["fraud_percentage"])
    print("Identity coverage:", metadata["identity_coverage_percentage"])
    print("Timestamp range:", metadata["timestamp_min"], "to", metadata["timestamp_max"])
    print("Parquet file:", OUTPUT_FILE)
    print("Metadata file:", METADATA_FILE)


if __name__ == "__main__":
    prepare_dataset()
