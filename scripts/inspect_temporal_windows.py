
"""Inspect chronological windows for temporal adaptation experiments."""

import pandas as pd

VALIDATION_PATH = "data/processed/splits/validation.parquet"
TEST_PATH = "data/processed/splits/test.parquet"


def describe_window(name, df):
    return {
        "window": name,
        "rows": len(df),
        "fraud_count": int(df["isFraud"].sum()),
        "legitimate_count": int((df["isFraud"] == 0).sum()),
        "fraud_percentage": round(
            df["isFraud"].mean() * 100, 4
        ),
        "timestamp_min": int(df["TransactionDT"].min()),
        "timestamp_max": int(df["TransactionDT"].max()),
    }


def main():
    print("Loading validation partition...")

    validation = pd.read_parquet(VALIDATION_PATH)
    validation = validation.sort_values(
        "TransactionDT"
    ).reset_index(drop=True)

    test = pd.read_parquet(TEST_PATH)

    # Divide validation into three approximately equal
    # chronological windows.
    windows = validation["TransactionDT"].quantile(
        [1 / 3, 2 / 3]
    ).tolist()

    boundary_1, boundary_2 = windows

    initial = validation[
        validation["TransactionDT"] <= boundary_1
    ]

    adaptation = validation[
        (validation["TransactionDT"] > boundary_1)
        & (validation["TransactionDT"] <= boundary_2)
    ]

    evaluation = validation[
        validation["TransactionDT"] > boundary_2
    ]

    # Report each window.
    reports = [
        describe_window("initial_calibration", initial),
        describe_window("adaptation", adaptation),
        describe_window("evaluation", evaluation),
    ]

    report_df = pd.DataFrame(reports)

    print("\nTEMPORAL WINDOW REPORT")
    print(report_df.to_string(index=False))

    # Verify chronology.
    assert (
        initial["TransactionDT"].max()
        < adaptation["TransactionDT"].min()
    )

    assert (
        adaptation["TransactionDT"].max()
        < evaluation["TransactionDT"].min()
    )

    assert len(initial) + len(adaptation) + len(evaluation) == len(
        validation
    )

    # Verify that the original final test partition is untouched.
    assert (
        evaluation["TransactionDT"].max()
        < test["TransactionDT"].min()
    )

    print("\nTEMPORAL VALIDATION")
    print("Initial -> Adaptation: valid")
    print("Adaptation -> Evaluation: valid")
    print("Evaluation -> Final test: valid")
    print("All validation rows accounted for.")

    print("\nProposed boundaries:")
    print(f"Boundary 1: {boundary_1}")
    print(f"Boundary 2: {boundary_2}")

    output_path = (
        "experiments/results/temporal_window_metadata.csv"
    )
    report_df.to_csv(output_path, index=False)

    print(f"\nMetadata saved to: {output_path}")


if __name__ == "__main__":
    main()
