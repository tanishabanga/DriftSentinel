import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_feature_drift import evaluate

ROOT = Path(".")
PROFILE = ROOT / "artifacts/monitoring/reference_profile.json"
SOURCE = ROOT / "data/processed/splits/validation.parquet"
OUT = ROOT / "artifacts/monitoring/validation"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(SOURCE).copy()

with PROFILE.open(encoding="utf-8") as file:
    profile = json.load(file)

# Scenario 1: unmodified data.
baseline_path = OUT / "baseline.parquet"
df.to_parquet(baseline_path, index=False)

# Scenario 2: deliberately shift TransactionAmt.
numeric_shift = df.copy()
numeric_shift["TransactionAmt"] = (
    pd.to_numeric(numeric_shift["TransactionAmt"], errors="coerce") * 3
)
numeric_path = OUT / "numeric_shift.parquet"
numeric_shift.to_parquet(numeric_path, index=False)

# Scenario 3: deliberately increase DeviceInfo missingness.
missing_shift = df.copy()
rng = np.random.default_rng(42)
mask = rng.random(len(missing_shift)) < 0.5
missing_shift.loc[mask, "DeviceInfo"] = np.nan

missing_path = OUT / "missingness_shift.parquet"
missing_shift.to_parquet(missing_path, index=False)

scenarios = [
    ("Baseline", baseline_path),
    ("Numeric shift", numeric_path),
    ("Missingness shift", missing_path),
]

results = {}

for name, path in scenarios:
    output = OUT / f"{name.lower().replace(' ', '_')}_report.csv"
    print(f"\n--- {name} ---")
    result = evaluate(path, PROFILE, output)
    results[name] = result.set_index("feature")["psi"]

print("\nCONTROLLED DRIFT VALIDATION")
print(
    "Baseline TransactionAmt PSI:",
    f"{results['Baseline']['TransactionAmt']:.6f}",
)
print(
    "Shifted TransactionAmt PSI:",
    f"{results['Numeric shift']['TransactionAmt']:.6f}",
)
print(
    "Baseline DeviceInfo PSI:",
    f"{results['Baseline']['DeviceInfo']:.6f}",
)
print(
    "Shifted DeviceInfo PSI:",
    f"{results['Missingness shift']['DeviceInfo']:.6f}",
)

assert results["Numeric shift"]["TransactionAmt"] > results["Baseline"]["TransactionAmt"]
assert results["Missingness shift"]["DeviceInfo"] > results["Baseline"]["DeviceInfo"]

print("\nPASS: Both controlled changes increased their corresponding PSI.")
