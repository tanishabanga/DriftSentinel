
"""Train a model on the matched-size temporal dataset."""

import json
import time
from pathlib import Path

import pandas as pd
from joblib import dump

from driftsentinel.models.baseline import build_logistic_regression_pipeline

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/temporal/matched_size_train.parquet"
MODEL_PATH = ROOT / "artifacts/temporal/matched_size_model.joblib"
METADATA_PATH = ROOT / "experiments/results/matched_size_model_metadata.json"

TARGET = "isFraud"
EXCLUDED = ["TransactionID", "TransactionDT", TARGET]


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    if df.empty:
        raise ValueError("Training dataset is empty.")

    if df["TransactionID"].duplicated().any():
        raise ValueError("Duplicate transaction IDs detected.")

    X = df.drop(columns=EXCLUDED)
    y = df[TARGET]

    if y.nunique() != 2:
        raise ValueError("Both classes must be present.")

    model = build_logistic_regression_pipeline(X)

    print("Matched-size model training")
    print(f"Rows: {len(df):,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print(f"Fraud rate: {y.mean():.4%}")
    print(f"Features: {X.shape[1]:,}")

    start = time.perf_counter()
    model.fit(X, y)
    training_seconds = time.perf_counter() - start

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    dump(model, MODEL_PATH)

    metadata = {
        "experiment": "matched_size_temporal_model",
        "training_rows": len(df),
        "fraud_cases": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "feature_count": X.shape[1],
        "training_start": int(df["TransactionDT"].min()),
        "training_end": int(df["TransactionDT"].max()),
        "training_seconds": training_seconds,
        "model_type": "LogisticRegression",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print(f"\nTraining completed in {training_seconds:.2f} seconds.")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")


if __name__ == "__main__":
    main()
