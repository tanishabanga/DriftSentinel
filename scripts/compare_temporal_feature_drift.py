from pathlib import Path

import pandas as pd

root = Path("artifacts/monitoring/temporal")

windows = {
    "initial": "initial_calibration_psi.csv",
    "adaptation": "adaptation_psi.csv",
    "evaluation": "evaluation_psi.csv",
}

merged = None

for window, filename in windows.items():
    df = pd.read_csv(root / filename)

    df = df[[
        "feature",
        "feature_type",
        "psi",
        "reference_missing_rate",
        "current_missing_rate",
    ]].copy()

    df = df.rename(columns={
        "psi": f"{window}_psi",
        "current_missing_rate": f"{window}_missing_rate",
    })

    if merged is None:
        merged = df
    else:
        merged = merged.merge(
            df.drop(columns=["feature_type", "reference_missing_rate"]),
            on="feature",
            how="inner",
            validate="one_to_one",
        )

merged["mean_psi"] = merged[
    ["initial_psi", "adaptation_psi", "evaluation_psi"]
].mean(axis=1)

merged["psi_range"] = (
    merged[["initial_psi", "adaptation_psi", "evaluation_psi"]].max(axis=1)
    - merged[["initial_psi", "adaptation_psi", "evaluation_psi"]].min(axis=1)
)

merged["evaluation_change_from_adaptation"] = (
    merged["evaluation_psi"] - merged["adaptation_psi"]
)

merged["persistent_gt_010"] = (
    merged[["initial_psi", "adaptation_psi", "evaluation_psi"]] > 0.10
).all(axis=1)

merged["persistent_gt_025"] = (
    merged[["initial_psi", "adaptation_psi", "evaluation_psi"]] > 0.25
).all(axis=1)

output = root / "feature_temporal_comparison.csv"
merged.to_csv(output, index=False)

print("PERSISTENT FEATURES: PSI > 0.10 IN ALL WINDOWS")
print(
    merged[merged["persistent_gt_010"]]
    .sort_values("mean_psi", ascending=False)
    .head(20)
    [["feature", "initial_psi", "adaptation_psi", "evaluation_psi"]]
    .round(5)
    .to_string(index=False)
)

print("\nPERSISTENT FEATURES: PSI > 0.25 IN ALL WINDOWS")
print(
    merged[merged["persistent_gt_025"]]
    [["feature", "initial_psi", "adaptation_psi", "evaluation_psi"]]
    .round(5)
    .to_string(index=False)
)

print("\nLARGEST PSI INCREASE: ADAPTATION TO EVALUATION")
print(
    merged.sort_values(
        "evaluation_change_from_adaptation",
        ascending=False,
    ).head(15)[
        ["feature", "adaptation_psi", "evaluation_psi",
         "evaluation_change_from_adaptation"]
    ].round(5).to_string(index=False)
)

print("\nLARGEST PSI DECREASE: INITIAL TO ADAPTATION")
merged["adaptation_change"] = (
    merged["adaptation_psi"] - merged["initial_psi"]
)

print(
    merged.sort_values("adaptation_change").head(15)[
        ["feature", "initial_psi", "adaptation_psi", "adaptation_change"]
    ].round(5).to_string(index=False)
)

print("\nSUMMARY")
print("Features compared:", len(merged))
print("Persistent PSI > 0.10:", int(merged["persistent_gt_010"].sum()))
print("Persistent PSI > 0.25:", int(merged["persistent_gt_025"].sum()))
print("Saved:", output)
