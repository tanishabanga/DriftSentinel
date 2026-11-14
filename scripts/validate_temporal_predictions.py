
"""Validate prediction alignment across the four temporal models."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    ROOT / "experiments/results/temporal_predictions"
)

MODELS = [
    "frozen",
    "recent_only",
    "matched_size",
    "retrained",
]

WINDOWS = [
    "adaptation_calibration",
    "evaluation",
]

REQUIRED_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "probability",
}


def load_predictions(model: str, window: str) -> pd.DataFrame:
    filename = f"{window}_{model}_predictions.parquet"
    path = PREDICTION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")

    if df.empty:
        raise ValueError(f"{filename}: empty prediction file")

    if df["TransactionID"].duplicated().any():
        raise ValueError(f"{filename}: duplicate transaction IDs")

    if not df["probability"].between(0, 1).all():
        raise ValueError(f"{filename}: invalid probabilities")

    if not df["isFraud"].isin([0, 1]).all():
        raise ValueError(f"{filename}: invalid target labels")

    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(f"{filename}: timestamps are not ordered")

    return df


def validate_window(window: str) -> None:
    reference = load_predictions(MODELS[0], window)

    for model in MODELS[1:]:
        candidate = load_predictions(model, window)

        if not reference["TransactionID"].equals(
            candidate["TransactionID"]
        ):
            raise ValueError(
                f"{window}: transaction IDs differ for {model}"
            )

        if not reference["TransactionDT"].equals(
            candidate["TransactionDT"]
        ):
            raise ValueError(
                f"{window}: timestamps differ for {model}"
            )

        if not reference["isFraud"].equals(
            candidate["isFraud"]
        ):
            raise ValueError(
                f"{window}: target labels differ for {model}"
            )

    print(f"\n{window.upper()} WINDOW")
    print(f"Transactions: {len(reference):,}")
    print(f"Fraud cases: {int(reference['isFraud'].sum()):,}")
    print(f"Fraud rate: {reference['isFraud'].mean():.4%}")
    print(f"Timestamp range: "
          f"{reference['TransactionDT'].min()} - "
          f"{reference['TransactionDT'].max()}")
    print("All four models have identical transaction IDs, timestamps and labels.")


def main() -> None:
    for window in WINDOWS:
        validate_window(window)

    print("\nSUCCESS: Prediction alignment validation passed.")


if __name__ == "__main__":
    main()
