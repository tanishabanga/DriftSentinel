
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")
CHUNK_SIZE = 100_000


def inspect_transactions(path: Path) -> None:
    print(f"\nInspecting: {path.name}")

    total_rows = 0
    fraud_counts = {0: 0, 1: 0}
    missing_counts = None
    transaction_ids = set()
    timestamp_min = None
    timestamp_max = None

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

        fraud_counts_chunk = chunk["isFraud"].value_counts()
        for label, count in fraud_counts_chunk.items():
            fraud_counts[int(label)] += int(count)

        missing = chunk.isna().sum()

        if missing_counts is None:
            missing_counts = missing
        else:
            missing_counts = missing_counts.add(
                missing, fill_value=0
            )

        ids = chunk["TransactionID"]

        if ids.duplicated().any():
            raise ValueError("Duplicate IDs found within a chunk.")

        if any(int(value) in transaction_ids for value in ids):
            raise ValueError("Duplicate IDs found across chunks.")

        transaction_ids.update(ids.astype(int).tolist())

        current_min = chunk["TransactionDT"].min()
        current_max = chunk["TransactionDT"].max()

        timestamp_min = (
            current_min if timestamp_min is None
            else min(timestamp_min, current_min)
        )
        timestamp_max = (
            current_max if timestamp_max is None
            else max(timestamp_max, current_max)
        )

    print("Rows:", total_rows)
    print("Fraud counts:", fraud_counts)
    print(
        "Fraud percentage:",
        round(fraud_counts[1] / total_rows * 100, 4),
    )
    print("Timestamp range:", timestamp_min, "to", timestamp_max)
    print("Unique transaction IDs:", len(transaction_ids))

    missing_report = (
        missing_counts
        .sort_values(ascending=False)
        .rename("missing_count")
        .to_frame()
    )
    missing_report["missing_percentage"] = (
        missing_report["missing_count"] / total_rows * 100
    )

    print("\nTop 15 columns by missingness:")
    print(missing_report.head(15).to_string())


def inspect_identity(path: Path, transaction_path: Path) -> None:
    print(f"\nInspecting: {path.name}")

    identity = pd.read_csv(
        path,
        usecols=["TransactionID"],
    )

    transaction_ids = set()

    for chunk in pd.read_csv(
        transaction_path,
        usecols=["TransactionID"],
        chunksize=CHUNK_SIZE,
    ):
        transaction_ids.update(
            chunk["TransactionID"].astype(int).tolist()
        )

    identity_ids = identity["TransactionID"].astype(int)

    print("Identity rows:", len(identity))
    print("Unique identity IDs:", identity_ids.nunique())
    print("Duplicate identity IDs:", identity_ids.duplicated().sum())
    print(
        "Identity IDs found in transactions:",
        identity_ids.isin(transaction_ids).sum(),
    )


if __name__ == "__main__":
    train_path = DATA_DIR / "train_transaction.csv"
    identity_path = DATA_DIR / "train_identity.csv"

    inspect_transactions(train_path)
    inspect_identity(identity_path, train_path)
