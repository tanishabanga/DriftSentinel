import json
from pathlib import Path

import pandas as pd

SCHEMA_PATH = Path("artifacts/xgboost/feature_schema.json")
DATA_PATH = Path("data/processed/splits/validation.parquet")
OUTPUT_PATH = Path("artifacts/monitoring/validation/unseen_category_report.csv")

with SCHEMA_PATH.open(encoding="utf-8") as file:
    schema = json.load(file)

df = pd.read_parquet(DATA_PATH)

unseen_matrix = pd.DataFrame(index=df.index)
rows = []

for column in schema["categorical_columns"]:
    vocabulary = {
        str(value)
        for value in schema["categorical_vocabularies"][column]
    }

    observed = df[column].astype("string")
    unseen_mask = observed.notna() & ~observed.isin(vocabulary)

    unseen_matrix[column] = unseen_mask

    rows.append({
        "feature": column,
        "unseen_categories": int(
            observed[unseen_mask].nunique()
        ),
        "unseen_rows": int(unseen_mask.sum()),
        "unseen_row_rate": float(unseen_mask.mean()),
        "missing_rate": float(df[column].isna().mean()),
    })

report = pd.DataFrame(rows).sort_values(
    "unseen_row_rate",
    ascending=False,
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(OUTPUT_PATH, index=False)

affected_transactions = int(unseen_matrix.any(axis=1).sum())
total_transactions = len(df)

print("UNSEEN CATEGORY REPORT")
print(report.head(15).round(5).to_string(index=False))

print("\nTRANSACTION-LEVEL SUMMARY")
print(f"Total transactions: {total_transactions:,}")
print(f"Transactions with unseen categories: {affected_transactions:,}")
print(
    "Affected transaction rate:",
    f"{affected_transactions / total_transactions:.4%}",
)
print(f"Feature-row occurrences: {int(unseen_matrix.sum().sum()):,}")

print(f"\nSaved: {OUTPUT_PATH}")
