import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/processed/splits/validation.parquet")
SCHEMA_PATH = Path("artifacts/xgboost/feature_schema.json")
API_URL = "http://127.0.0.1:8000/monitoring/snapshots"
BATCH_SIZE = 1000

frame = pd.read_parquet(DATA_PATH).iloc[BATCH_SIZE:2 * BATCH_SIZE].copy()

with SCHEMA_PATH.open("r", encoding="utf-8") as file:
    schema = json.load(file)

feature_names = schema["feature_names"]

if len(frame) != BATCH_SIZE:
    raise ValueError(f"Expected {BATCH_SIZE} rows, found {len(frame)}.")

if len(feature_names) != 431:
    raise ValueError(f"Expected 431 schema features, found {len(feature_names)}.")

missing = [name for name in feature_names if name not in frame.columns]
if missing:
    raise ValueError(f"Missing model features: {missing}")

if "TransactionID" not in frame.columns:
    raise ValueError("TransactionID is missing.")

# Select only the model's exact feature schema.
# Keep TransactionID as metadata; exclude TransactionDT and isFraud.
request_frame = frame.loc[:, ["TransactionID", *feature_names]].copy()

if "isFraud" in request_frame.columns or "TransactionDT" in request_frame.columns:
    raise AssertionError("Non-model columns unexpectedly entered the request.")

payload = {
    "transactions": json.loads(request_frame.to_json(orient="records")),
    "metadata": {
        "source": "IEEE-CIS validation split",
        "evaluation_mode": "offline_validation_backtest",
        "batch_size": BATCH_SIZE,
        "ground_truth_labels_in_request": False,
        "relative_time_in_request": False,
        "purpose": "Second end-to-end monitoring integration test using a distinct validation batch",
    },
}

body = json.dumps(payload, allow_nan=False).encode("utf-8")

request = urllib.request.Request(
    API_URL,
    data=body,
    headers={"Content-Type": "application/json", "Accept": "application/json"},
    method="POST",
)

print(f"Submitting {len(request_frame)} transactions...")
print(f"Model features: {len(feature_names)}")
print("TransactionID included: Yes")
print("TransactionDT included: No")
print("Ground-truth labels included: No")

try:
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    print(f"API returned HTTP {exc.code}:")
    print(exc.read().decode("utf-8"))
    raise
except urllib.error.URLError as exc:
    raise RuntimeError(
        "Could not reach the monitoring API. "
        "Check that FastAPI is running on port 8000."
    ) from exc

snapshot = result["snapshot"]
prediction_summary = snapshot["prediction_summary"]
drift_summary = snapshot["drift_summary"]

print("\n===== SNAPSHOT CREATED =====")
print("Snapshot ID:", snapshot["snapshot_id"])
print("Prediction count:", snapshot["prediction_count"])
print("Reference rows:", snapshot["metadata"]["reference_row_count"])
print("Reference features:", snapshot["metadata"]["reference_feature_count"])

print("\n===== PREDICTION SUMMARY =====")
print("Actions:", prediction_summary["action_counts"])
print("Mean calibrated probability:", prediction_summary["calibrated_probability"]["mean"])
print("Mean expected cost:", prediction_summary["expected_cost"]["mean"])

print("\n===== FEATURE DRIFT SUMMARY =====")
print("Features evaluated:", drift_summary["feature_count"])
print("Mean PSI:", drift_summary["mean_psi"])
print("Features with PSI > 0.10:", drift_summary["features_psi_above_0_10"])
print("Features with PSI > 0.25:", drift_summary["features_psi_above_0_25"])

print("\n===== TOP 5 DRIFTING FEATURES =====")
for feature in drift_summary["features"][:5]:
    print(f'{feature["feature"]}: PSI={feature["psi"]:.6f}')

print("\nSnapshot successfully created and persisted.")
