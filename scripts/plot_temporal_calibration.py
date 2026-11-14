"""Plot reliability diagrams for all temporal calibration strategies."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

from driftsentinel.calibration.calibrator import ProbabilityCalibrator

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    ROOT / "experiments" / "results" / "temporal_predictions"
)

RESULT_DIR = ROOT / "experiments" / "results"

OUTPUT_PATH = RESULT_DIR / "temporal_calibration_reliability.png"
METRICS_PATH = RESULT_DIR / "temporal_calibration_bins.csv"

def load_predictions(filename: str) -> pd.DataFrame:
    """Load and validate a temporal prediction file."""

    path = PREDICTION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_parquet(path)

    required = {
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "probability",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {filename}: {missing}"
        )

    if df["TransactionID"].duplicated().any():
        raise ValueError(
            f"Duplicate transaction IDs in {filename}."
        )

    if not df["probability"].between(0, 1).all():
        raise ValueError(
            f"Invalid probabilities in {filename}."
        )

    return df

def validate_same_transactions(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> None:
    """Verify identical transaction IDs, timestamps and labels."""

    columns = ["TransactionID", "TransactionDT", "isFraud"]

    reference = (
        reference[columns]
        .sort_values("TransactionID")
        .reset_index(drop=True)
    )

    candidate = (
        candidate[columns]
        .sort_values("TransactionID")
        .reset_index(drop=True)
    )

    if not reference.equals(candidate):
        raise ValueError(
            "Prediction files contain different transactions, "
            "timestamps, or labels."
        )


def build_strategies():
    """Reconstruct all 14 temporal calibration strategies."""

    # Load calibration predictions.
    initial = load_predictions(
        "initial_calibration_predictions.parquet"
    )

    adaptation_frozen = load_predictions(
        "adaptation_calibration_frozen_predictions.parquet"
    )
    adaptation_retrained = load_predictions(
        "adaptation_calibration_retrained_predictions.parquet"
    )
    adaptation_recent = load_predictions(
        "adaptation_calibration_recent_only_predictions.parquet"
    )
    adaptation_matched = load_predictions(
        "adaptation_calibration_matched_size_predictions.parquet"
    )

    # Load evaluation predictions.
    evaluation_frozen = load_predictions(
        "evaluation_frozen_predictions.parquet"
    )
    evaluation_retrained = load_predictions(
        "evaluation_retrained_predictions.parquet"
    )
    evaluation_recent = load_predictions(
        "evaluation_recent_only_predictions.parquet"
    )
    evaluation_matched = load_predictions(
        "evaluation_matched_size_predictions.parquet"
    )

    # Verify calibration-window alignment.
    for candidate in [
        adaptation_retrained,
        adaptation_recent,
        adaptation_matched,
    ]:
        validate_same_transactions(adaptation_frozen, candidate)

    # Verify evaluation-window alignment.
    for candidate in [
        evaluation_retrained,
        evaluation_recent,
        evaluation_matched,
    ]:
        validate_same_transactions(evaluation_frozen, candidate)

    y_initial = initial["isFraud"].to_numpy()
    y_adaptation = adaptation_frozen["isFraud"].to_numpy()
    y_evaluation = evaluation_frozen["isFraud"].to_numpy()

    # Raw calibration-window probabilities.
    p_initial = initial["probability"].to_numpy()

    p_adaptation_frozen = adaptation_frozen["probability"].to_numpy()
    p_adaptation_retrained = adaptation_retrained["probability"].to_numpy()
    p_adaptation_recent = adaptation_recent["probability"].to_numpy()
    p_adaptation_matched = adaptation_matched["probability"].to_numpy()

    # Raw evaluation probabilities.
    p_evaluation_frozen = evaluation_frozen["probability"].to_numpy()
    p_evaluation_retrained = evaluation_retrained["probability"].to_numpy()
    p_evaluation_recent = evaluation_recent["probability"].to_numpy()
    p_evaluation_matched = evaluation_matched["probability"].to_numpy()

    # Static calibrators: fit on initial calibration window.
    static_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_initial, p_initial
    )
    static_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_initial, p_initial
    )

    # Adaptive calibrators: frozen model, adaptation window.
    adaptive_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_adaptation, p_adaptation_frozen
    )
    adaptive_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation, p_adaptation_frozen
    )

    # Retrained model calibrators.
    retrained_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_adaptation, p_adaptation_retrained
    )
    retrained_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation, p_adaptation_retrained
    )

    # Recent-only model calibrators.
    recent_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_adaptation, p_adaptation_recent
    )
    recent_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation, p_adaptation_recent
    )

    # Matched-size model calibrators.
    matched_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_adaptation, p_adaptation_matched
    )
    matched_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation, p_adaptation_matched
    )

    strategies = {
        "frozen_uncalibrated": p_evaluation_frozen,
        "frozen_static_sigmoid": static_sigmoid.predict(
            p_evaluation_frozen
        ),
        "frozen_static_isotonic": static_isotonic.predict(
            p_evaluation_frozen
        ),
        "frozen_adaptive_sigmoid": adaptive_sigmoid.predict(
            p_evaluation_frozen
        ),
        "frozen_adaptive_isotonic": adaptive_isotonic.predict(
            p_evaluation_frozen
        ),
        "retrained_uncalibrated": p_evaluation_retrained,
        "retrained_sigmoid": retrained_sigmoid.predict(
            p_evaluation_retrained
        ),
        "retrained_isotonic": retrained_isotonic.predict(
            p_evaluation_retrained
        ),
        "recent_only_uncalibrated": p_evaluation_recent,
        "recent_only_sigmoid": recent_sigmoid.predict(
            p_evaluation_recent
        ),
        "recent_only_isotonic": recent_isotonic.predict(
            p_evaluation_recent
        ),
        "matched_size_uncalibrated": p_evaluation_matched,
        "matched_size_sigmoid": matched_sigmoid.predict(
            p_evaluation_matched
        ),
        "matched_size_isotonic": matched_isotonic.predict(
            p_evaluation_matched
        ),
    }

    return y_evaluation, strategies

def plot_reliability_panel(
    ax,
    y_true,
    strategies,
    strategy_names,
    group_name,
):
    """Plot reliability curves and collect quantile-bin diagnostics."""

    bin_records = []

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        alpha=0.6,
        label="Perfect calibration",
    )

    for name in strategy_names:
        probabilities = strategies[name]

        fraction_positive, mean_predicted = calibration_curve(
            y_true,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )

        ax.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            markersize=4,
            label=name.replace(group_name + "_", ""),
        )

        # Preserve the underlying quantile-bin diagnostics.
        bin_data = pd.DataFrame({
            "probability": probabilities,
            "actual": y_true,
        })

        bin_data["bin"] = pd.qcut(
            bin_data["probability"],
            q=10,
            labels=False,
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

        bin_summary["strategy"] = name
        bin_summary["group"] = group_name

        bin_records.extend(bin_summary.to_dict("records"))

    ax.set_title(group_name.replace("_", " ").title())
    ax.set_xlabel("Mean predicted fraud probability")
    ax.set_ylabel("Observed fraud frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")

    return bin_records

def main():
    """Generate temporal reliability diagrams and bin diagnostics."""

    y_evaluation, strategies = build_strategies()

    strategy_groups = {
        "frozen": [
            "frozen_uncalibrated",
            "frozen_static_sigmoid",
            "frozen_static_isotonic",
            "frozen_adaptive_sigmoid",
            "frozen_adaptive_isotonic",
        ],
        "retrained": [
            "retrained_uncalibrated",
            "retrained_sigmoid",
            "retrained_isotonic",
        ],
        "recent_only": [
            "recent_only_uncalibrated",
            "recent_only_sigmoid",
            "recent_only_isotonic",
        ],
        "matched_size": [
            "matched_size_uncalibrated",
            "matched_size_sigmoid",
            "matched_size_isotonic",
        ],
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12),
    )

    all_bin_records = []

    for ax, (group_name, strategy_names) in zip(
        axes.flat,
        strategy_groups.items(),
    ):
        records = plot_reliability_panel(
            ax=ax,
            y_true=y_evaluation,
            strategies=strategies,
            strategy_names=strategy_names,
            group_name=group_name,
        )

        all_bin_records.extend(records)

    fig.suptitle(
        "Temporal Probability Calibration Comparison",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    bin_results = pd.DataFrame(all_bin_records)
    bin_results.to_csv(METRICS_PATH, index=False)

    print("\nTEMPORAL CALIBRATION AUDIT")
    print("=" * 65)
    print(f"Evaluation transactions: {len(y_evaluation):,}")
    print(f"Evaluation fraud cases: {int(y_evaluation.sum()):,}")
    print(f"Fraud rate: {y_evaluation.mean():.4%}")
    print(f"Strategies evaluated: {len(strategies)}")
    print(f"Reliability plot: {OUTPUT_PATH}")
    print(f"Bin diagnostics: {METRICS_PATH}")
    print(f"Diagnostic rows: {len(bin_results)}")


if __name__ == "__main__":
    main()