
"""Score frozen and retrained models on shared temporal windows."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "temporal"
ARTIFACT_DIR = ROOT / "artifacts" / "temporal"

OUTPUT_DIR = (
    ROOT / "experiments" / "results" / "temporal_predictions"
)

MODELS = {
    "frozen": ARTIFACT_DIR / "baseline_model.joblib",
    "retrained": ARTIFACT_DIR / "retrained_model.joblib",
}

WINDOWS = [
    "adaptation_calibration",
    "evaluation",
]


def score_model(model, df: pd.DataFrame) -> pd.DataFrame:
    """Generate positive-class probabilities for a temporal window."""

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

    if not output["probability"].between(0, 1).all():
        raise ValueError("Model produced invalid probabilities.")

    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {}

    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        print(f"Loading {name} model...")
        models[name] = joblib.load(path)

    for window in WINDOWS:
        print(f"\nLoading {window}...")
        df = pd.read_parquet(DATA_DIR / f"{window}.parquet")

        print(f"Rows: {len(df):,}")

        for model_name, model in models.items():
            print(f"Scoring with {model_name} model...")

            output = score_model(model, df)

            output_path = (
                OUTPUT_DIR
                / f"{window}_{model_name}_predictions.parquet"
            )

            output.to_parquet(output_path, index=False)

            print(
                f"Mean predicted probability: "
                f"{output['probability'].mean():.6f}"
            )
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
