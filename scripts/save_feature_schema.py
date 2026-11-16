
"""Generate the saved XGBoost inference feature schema."""

from pathlib import Path

from driftsentinel.models.feature_schema import (
    build_feature_schema,
    save_feature_schema,
)

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/splits/train.parquet"

OUTPUT_PATH = (
    ROOT / "artifacts/xgboost/feature_schema.json"
)


def main() -> None:
    print("Building feature schema from training data...")

    schema = build_feature_schema(TRAIN_PATH)

    save_feature_schema(schema, OUTPUT_PATH)

    print("\nFEATURE SCHEMA")
    print("Feature count:", schema["feature_count"])
    print("Numerical features:", len(schema["numerical_columns"]))
    print("Categorical features:", len(schema["categorical_columns"]))
    print("Categorical vocabularies:",
          len(schema["categorical_vocabularies"]))

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
