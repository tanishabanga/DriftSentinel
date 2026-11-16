"""Run five-fold rolling fraud detection and calibration experiments."""

from pathlib import Path
from time import perf_counter

import pandas as pd

from driftsentinel.calibration.calibrator import ProbabilityCalibrator
from driftsentinel.calibration.metrics import evaluate_calibration
from driftsentinel.experiments.temporal import create_rolling_folds
from driftsentinel.models.baseline import (
    EXCLUDED_COLUMNS,
    build_logistic_regression_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/splits/train.parquet"
VALIDATION_PATH = ROOT / "data/processed/splits/validation.parquet"

PREDICTION_DIR = ROOT / "experiments/results/rolling_predictions"
RESULTS_DIR = ROOT / "experiments/results"

N_FOLDS = 5
CALIBRATION_METHODS = ("sigmoid", "isotonic")


def load_development_data():
    """Load development partitions and create initial historical windows."""

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    required = set(EXCLUDED_COLUMNS)

    for name, df in (("train", train), ("validation", validation)):
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    if train.empty or validation.empty:
        raise ValueError("Development partitions cannot be empty.")

    # Reserve the most recent 20% of training history for initial calibration.
    boundary = train["TransactionDT"].quantile(0.80)

    initial_train = train.loc[train["TransactionDT"] <= boundary].copy()

    initial_calibration = train.loc[train["TransactionDT"] > boundary].copy()

    if initial_train.empty or initial_calibration.empty:
        raise ValueError("Initial training/calibration split is empty.")

    return initial_train, initial_calibration, validation


def positive_class_probabilities(model, df):
    """Generate probabilities for the fraud class (label 1)."""

    X = df.drop(columns=EXCLUDED_COLUMNS)

    probabilities = model.predict_proba(X)
    classes = model.named_steps["classifier"].classes_

    positive_indices = [index for index, label in enumerate(classes) if label == 1]

    if len(positive_indices) != 1:
        raise ValueError("Model does not contain exactly one positive class.")

    result = probabilities[:, positive_indices[0]]

    if not pd.Series(result).between(0, 1).all():
        raise ValueError("Model generated invalid probabilities.")

    return result


def fit_calibrators(y_true, probabilities):
    """Fit sigmoid and isotonic calibration mappings."""

    calibrators = {}

    for method in CALIBRATION_METHODS:
        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(y_true, probabilities)
        calibrators[method] = calibrator

    return calibrators


def evaluate_strategies(y_true, strategies):
    """Calculate calibration and ranking metrics for each strategy."""

    rows = []

    for strategy_name, probabilities in strategies.items():
        metrics = evaluate_calibration(y_true, probabilities)

        rows.append(
            {
                "strategy": strategy_name,
                **metrics,
            }
        )

    return rows


def run_fold(fold, initial_calibration):
    """Train and evaluate all calibration strategies for one fold."""

    print(f"\n{'=' * 65}")
    print(f"FOLD {fold.fold_id}")
    print(f"{'=' * 65}")

    train_df = fold.adaptation_train
    calibration_df = fold.adaptation_calibration
    evaluation_df = fold.evaluation

    print(f"Training rows:     {len(train_df):,}")
    print(f"Static calibration: {len(initial_calibration):,}")
    print(f"Adaptive calibration: {len(calibration_df):,}")
    print(f"Evaluation rows:   {len(evaluation_df):,}")

    if train_df["isFraud"].nunique() != 2:
        raise ValueError(f"Fold {fold.fold_id}: training needs both classes.")

    # Train one fresh model for this fold.
    X_train = train_df.drop(columns=EXCLUDED_COLUMNS)
    y_train = train_df["isFraud"].astype(int)

    model = build_logistic_regression_pipeline(X_train)

    print("Fitting preprocessing and Logistic Regression...", flush=True)

    start = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - start

    print(
        f"Model training completed in {training_seconds:.2f} seconds",
        flush=True,
    )

    # Generate predictions using the same fold-specific base model.
    static_probabilities = positive_class_probabilities(model, initial_calibration)

    adaptive_probabilities = positive_class_probabilities(model, calibration_df)

    evaluation_probabilities = positive_class_probabilities(model, evaluation_df)

    static_calibrators = fit_calibrators(
        initial_calibration["isFraud"],
        static_probabilities,
    )

    if fold.fold_id == 1:
        # Fold 1 uses the same historical data for both strategies.
        adaptive_calibrators = static_calibrators
    else:
        adaptive_calibrators = fit_calibrators(
            calibration_df["isFraud"],
            adaptive_probabilities,
        )

    # Build strategies for every fold.
    strategies = {
        "uncalibrated": evaluation_probabilities,
        "static_sigmoid": static_calibrators["sigmoid"].predict(
            evaluation_probabilities
        ),
        "static_isotonic": static_calibrators["isotonic"].predict(
            evaluation_probabilities
        ),
        "adaptive_sigmoid": adaptive_calibrators["sigmoid"].predict(
            evaluation_probabilities
        ),
        "adaptive_isotonic": adaptive_calibrators["isotonic"].predict(
            evaluation_probabilities
        ),
    }

    # Preserve transaction-level predictions for later analysis.
    predictions = evaluation_df[["TransactionID", "TransactionDT", "isFraud"]].copy()

    for name, probabilities in strategies.items():
        predictions[f"probability_{name}"] = probabilities

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    prediction_path = PREDICTION_DIR / f"fold_{fold.fold_id}_predictions.parquet"

    predictions.to_parquet(prediction_path, index=False)

    # Evaluate every strategy on the same evaluation transactions.
    metric_rows = evaluate_strategies(
        evaluation_df["isFraud"],
        strategies,
    )

    for row in metric_rows:
        row.update(
            {
                "fold_id": fold.fold_id,
                "training_rows": len(train_df),
                "static_calibration_rows": len(initial_calibration),
                "adaptive_calibration_rows": len(calibration_df),
                "evaluation_rows": len(evaluation_df),
                "training_seconds": round(training_seconds, 4),
            }
        )

    print(f"Saved predictions: {prediction_path}")

    return metric_rows


def main():
    print("Loading development data...")

    initial_train, initial_calibration, validation = load_development_data()

    print(f"Initial training rows: {len(initial_train):,}")
    print(f"Initial calibration rows: {len(initial_calibration):,}")
    print(f"Validation rows: {len(validation):,}")

    print("\nConstructing rolling folds...")

    experiment = create_rolling_folds(
        initial_train=initial_train,
        initial_calibration=initial_calibration,
        validation=validation,
        n_folds=N_FOLDS,
    )

    print(f"Constructed {len(experiment.folds)} folds.")

    all_results = []

    for fold in experiment.folds:
        fold_results = run_fold(
            fold=fold,
            initial_calibration=experiment.initial_calibration,
        )

        all_results.extend(fold_results)

    results_df = pd.DataFrame(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_results_path = RESULTS_DIR / "rolling_fold_metrics.csv"
    results_df.to_csv(fold_results_path, index=False)

    summary = (
        results_df.groupby("strategy")
        .agg(
            {
                "brier_score": ["mean", "std"],
                "log_loss": ["mean", "std"],
                "ece": ["mean", "std"],
                "average_precision": ["mean", "std"],
            }
        )
        .round(6)
    )

    summary_path = RESULTS_DIR / "rolling_experiment_summary.csv"
    summary.to_csv(summary_path)

    print("\nFOLD-LEVEL RESULTS")
    print(results_df.to_string(index=False))

    print("\nFIVE-FOLD STRATEGY SUMMARY")
    print(summary.to_string())

    print(f"\nFold metrics saved: {fold_results_path}")
    print(f"Summary saved: {summary_path}")
    print(f"Predictions saved in: {PREDICTION_DIR}")


if __name__ == "__main__":
    main()
