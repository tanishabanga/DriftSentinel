
"""Score the matched-size model on shared temporal windows."""

from pathlib import Path

import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "artifacts/temporal/matched_size_model.joblib"
DATA_DIR = ROOT / "data/processed/temporal"
OUTPUT_DIR = ROOT / "experiments/results/temporal_predictions"

WINDOWS = ["adaptation_calibration", "evaluation"]

TARGET = "isFraud"
EXCLUDED = ["TransactionID", "TransactionDT", TARGET]


def main() -> None:
    model = load(MODEL_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for window in WINDOWS:
        df = pd.read_parquet(DATA_DIR / f"{window}.parquet")

        X = df.drop(columns=EXCLUDED)
        probabilities = model.predict_proba(X)[:, 1]

        if not pd.Series(probabilities).between(0, 1).all():
            raise ValueError(f"Invalid probabilities in {window}.")

        predictions = df[
            ["TransactionID", "TransactionDT", TARGET]
        ].copy()

        predictions["probability"] = probabilities

        output_path = (
            OUTPUT_DIR / f"{window}_matched_size_predictions.parquet"
        )

        predictions.to_parquet(output_path, index=False)

        print(f"\nWindow: {window}")
        print(f"Rows: {len(predictions):,}")
        print(f"Fraud cases: {int(predictions[TARGET].sum()):,}")
        print(f"Mean predicted probability: {probabilities.mean():.6f}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
