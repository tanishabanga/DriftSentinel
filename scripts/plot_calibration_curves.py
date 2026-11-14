
"""Plot reliability diagrams for calibration experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

from driftsentinel.calibration.calibrator import ProbabilityCalibrator


def main():
    input_path = Path(
        "experiments/results/logistic_regression_pilot_predictions.parquet"
    )

    output_path = Path(
        "experiments/results/calibration_reliability.png"
    )

    df = pd.read_parquet(input_path)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Use the same chronological calibration/evaluation split.
    calibration_df = df.iloc[:2500]
    evaluation_df = df.iloc[2500:]

    y_calibration = calibration_df["isFraud"].to_numpy()
    p_calibration = calibration_df["probability"].to_numpy()

    y_evaluation = evaluation_df["isFraud"].to_numpy()
    p_evaluation = evaluation_df["probability"].to_numpy()

    # Fit calibrators using only the calibration segment.
    sigmoid = ProbabilityCalibrator("sigmoid")
    sigmoid.fit(y_calibration, p_calibration)

    isotonic = ProbabilityCalibrator("isotonic")
    isotonic.fit(y_calibration, p_calibration)

    probabilities = {
        "Uncalibrated": p_evaluation,
        "Sigmoid": sigmoid.predict(p_evaluation),
        "Isotonic": isotonic.predict(p_evaluation),
    }

    plt.figure(figsize=(9, 7))

    # Perfect calibration reference.
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for name, predicted in probabilities.items():
        fraction_positive, mean_predicted = calibration_curve(
            y_evaluation,
            predicted,
            n_bins=10,
            strategy="quantile",
        )
        bin_data = pd.DataFrame({
            "probability": predicted,
            "actual": y_evaluation,
        })

        bin_data["bin"] = pd.qcut(
            bin_data["probability"],
            q=10,
            duplicates="drop",
        )

        bin_summary = (
            bin_data.groupby("bin", observed=True)
            .agg(
                transaction_count=("actual", "size"),
                mean_probability=("probability", "mean"),
                observed_fraud_rate=("actual", "mean"),
            )
            .reset_index()
        )

        print(f"\n{name} — BIN DIAGNOSTICS")
        print(bin_summary.to_string(index=False))
        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=name,
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraud frequency")
    plt.title("Probability Calibration Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Calibration plot saved to: {output_path}")


if __name__ == "__main__":
    main()
