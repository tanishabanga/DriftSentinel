
"""Generate frozen baseline predictions for temporal windows."""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("artifacts/temporal/baseline_model.joblib")
DATA_DIR = Path("data/processed/temporal")
OUTPUT_DIR = Path("experiments/results/temporal_predictions")

WINDOWS = [
    "initial_calibration",
    "adaptation",
    "evaluation",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading frozen baseline model...")
    model = joblib.load(MODEL_PATH)

    for window in WINDOWS:
        print(f"\nScoring {window}...")

        df = pd.read_parquet(DATA_DIR / f"{window}.parquet")

        y = df["isFraud"]
        X = df.drop(
            columns=["isFraud", "TransactionID", "TransactionDT"]
        )

        probabilities = model.predict_proba(X)

        classes = model.named_steps["classifier"].classes_
        positive_index = list(classes).index(1)

        output = df[
            ["TransactionID", "TransactionDT", "isFraud"]
        ].copy()

        output["probability"] = probabilities[:, positive_index]

        output_path = OUTPUT_DIR / f"{window}_predictions.parquet"
        output.to_parquet(output_path, index=False)

        print(f"Rows scored: {len(output):,}")
        print(f"Fraud rate: {y.mean() * 100:.4f}%")
        print(
            "Mean predicted probability: "
            f"{output['probability'].mean():.6f}"
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
