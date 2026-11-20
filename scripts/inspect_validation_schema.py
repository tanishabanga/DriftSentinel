import json
import pandas as pd
from pathlib import Path

data_path = Path("data/processed/splits/validation.parquet")
schema_path = Path("artifacts/xgboost/feature_schema.json")

frame = pd.read_parquet(data_path)

with schema_path.open("r", encoding="utf-8") as file:
    schema = json.load(file)

print("===== VALIDATION DATA =====")
print("Total columns:", len(frame.columns))
print("Contains TransactionID:", "TransactionID" in frame.columns)
print("Contains isFraud:", "isFraud" in frame.columns)
print("Columns after removing isFraud:", len(frame.drop(columns=["isFraud"], errors="ignore").columns))

print("\n===== FEATURE SCHEMA =====")
print("Schema keys:", list(schema.keys()))

# Inspect common schema representations.
for key, value in schema.items():
    if isinstance(value, list):
        print(f"{key}: {len(value)} entries")
    elif isinstance(value, dict):
        print(f"{key}: dictionary with {len(value)} keys")

print("\n===== COLUMN NAMES NOT IN MODEL SCHEMA =====")
feature_names = schema.get("feature_names")

if feature_names is None:
    for key in ("features", "numerical_features", "numeric_features"):
        if isinstance(schema.get(key), list):
            print(f"Found candidate schema field: {key}")

    raise ValueError(
        "Could not identify feature_names automatically. "
        "Inspect the schema keys printed above."
    )

extras = [
    column for column in frame.columns
    if column not in feature_names and column != "isFraud"
]

missing = [
    column for column in feature_names
    if column not in frame.columns
]

print("Model feature count:", len(feature_names))
print("Extra columns:", extras)
print("Missing model features:", missing)
