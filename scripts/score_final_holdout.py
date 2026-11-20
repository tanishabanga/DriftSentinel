from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from driftsentinel.calibration.calibrator import ProbabilityCalibrator


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data/processed/splits/test.parquet"
ARTIFACT_DIR = ROOT / "artifacts/temporal"
PREDICTION_DIR = ROOT / "experiments/results/temporal_predictions"

CALIBRATION_DIR = (
    ROOT / "experiments/results/temporal_predictions"
)

OUTPUT_DIR = ROOT / "experiments/results/final_holdout_predictions"
OUTPUT_PATH = OUTPUT_DIR / "final_holdout_probabilities.parquet"
METADATA_PATH = OUTPUT_DIR / "final_holdout_metadata.json"


MODEL_PATHS = {
    "frozen": ARTIFACT_DIR / "baseline_model.joblib",
    "retrained": ARTIFACT_DIR / "retrained_model.joblib",
    "recent_only": ARTIFACT_DIR / "recent_only_model.joblib",
    "matched_size": ARTIFACT_DIR / "matched_size_model.joblib",
}

CALIBRATION_FILES = {
    "initial_frozen": "initial_calibration_predictions.parquet",
    "adaptation_frozen": "adaptation_calibration_frozen_predictions.parquet",
    "adaptation_retrained": "adaptation_calibration_retrained_predictions.parquet",
    "adaptation_recent_only": "adaptation_calibration_recent_only_predictions.parquet",
    "adaptation_matched_size": "adaptation_calibration_matched_size_predictions.parquet",
}


def load_calibration_predictions(name):
    path = CALIBRATION_DIR / CALIBRATION_FILES[name]

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    required = {"isFraud", "probability"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{path.name} is empty.")

    if not df["probability"].between(0, 1).all():
        raise ValueError(f"{path.name} has invalid probabilities.")

    if not df["isFraud"].isin([0, 1]).all():
        raise ValueError(f"{path.name} has invalid labels.")

    if df["isFraud"].nunique() != 2:
        raise ValueError(f"{path.name} must contain both classes.")

    return df


def positive_class_probabilities(model, X):
    probabilities = model.predict_proba(X)
    classes = list(model.named_steps["classifier"].classes_)

    if 1 not in classes:
        raise ValueError("Model does not contain positive class 1.")

    positive_index = classes.index(1)
    result = probabilities[:, positive_index]

    if not np.isfinite(result).all():
        raise ValueError("Model generated non-finite probabilities.")

    if ((result < 0) | (result > 1)).any():
        raise ValueError("Model generated probabilities outside [0, 1].")

    return result


def fit_calibrators(calibration_data):
    fitted = {}

    # Static calibration: frozen model on initial calibration window.
    y_initial = calibration_data["initial_frozen"]["isFraud"]
    p_initial = calibration_data["initial_frozen"]["probability"]

    fitted["frozen_static_sigmoid"] = (
        ProbabilityCalibrator("sigmoid").fit(y_initial, p_initial)
    )
    fitted["frozen_static_isotonic"] = (
        ProbabilityCalibrator("isotonic").fit(y_initial, p_initial)
    )

    # Adaptive calibration: each model uses its own predictions
    # on the later adaptation-calibration window.
    for model_name in [
        "frozen",
        "retrained",
        "recent_only",
        "matched_size",
    ]:
        key = (
            "adaptation_frozen"
            if model_name == "frozen"
            else f"adaptation_{model_name}"
        )

        df = calibration_data[key]
        y = df["isFraud"]
        p = df["probability"]

        fitted[f"{model_name}_adaptive_sigmoid"] = (
            ProbabilityCalibrator("sigmoid").fit(y, p)
        )
        fitted[f"{model_name}_adaptive_isotonic"] = (
            ProbabilityCalibrator("isotonic").fit(y, p)
        )

    return fitted


def main():
    if OUTPUT_PATH.exists() or METADATA_PATH.exists():
        raise FileExistsError(
            "Final holdout output already exists. "
            "Do not overwrite it; inspect and explicitly archive "
            "the existing output before rerunning."
        )

    for path in MODEL_PATHS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    print("Loading existing model artifacts...")
    models = {
        name: joblib.load(path)
        for name, path in MODEL_PATHS.items()
    }

    expected_features = list(
        models["frozen"].feature_names_in_
    )

    for name, model in models.items():
        features = list(model.feature_names_in_)

        if features != expected_features:
            raise ValueError(
                f"{name} model feature order differs from frozen model."
            )

    print("Loading existing calibration predictions...")
    calibration_data = {
        name: load_calibration_predictions(name)
        for name in CALIBRATION_FILES
    }

    print("Fitting calibration mappings...")
    calibrators = fit_calibrators(calibration_data)

    # Explicitly exclude the target from the final holdout read.
    required_test_columns = [
        "TransactionID",
        "TransactionDT",
        *expected_features,
    ]

    print("Reading final holdout features without target labels...")
    test = pd.read_parquet(
        TEST_PATH,
        columns=required_test_columns,
    )

    if len(test) == 0:
        raise ValueError("Final holdout partition is empty.")

    if test["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionID values in final holdout.")

    if not test["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Final holdout timestamps are not chronological.")

    X_test = test[expected_features]

    print(f"Final holdout rows: {len(test):,}")
    print(f"Input features: {len(expected_features)}")

    print("Generating uncalibrated and calibrated probabilities...")
    output = test[["TransactionID", "TransactionDT"]].copy()

    base_probabilities = {}

    for name, model in models.items():
        print(f"Scoring {name}...")
        base_probabilities[name] = positive_class_probabilities(
            model,
            X_test,
        )

    # Frozen strategies.
    frozen = base_probabilities["frozen"]
    output["frozen_uncalibrated"] = frozen
    output["frozen_static_sigmoid"] = (
        calibrators["frozen_static_sigmoid"].predict(frozen)
    )
    output["frozen_static_isotonic"] = (
        calibrators["frozen_static_isotonic"].predict(frozen)
    )
    output["frozen_adaptive_sigmoid"] = (
        calibrators["frozen_adaptive_sigmoid"].predict(frozen)
    )
    output["frozen_adaptive_isotonic"] = (
        calibrators["frozen_adaptive_isotonic"].predict(frozen)
    )

    # Retrained, recent-only, and matched-size strategies.
    for name in ["retrained", "recent_only", "matched_size"]:
        probabilities = base_probabilities[name]

        output[f"{name}_uncalibrated"] = probabilities
        output[f"{name}_sigmoid"] = (
            calibrators[f"{name}_adaptive_sigmoid"].predict(
                probabilities
            )
        )
        output[f"{name}_isotonic"] = (
            calibrators[f"{name}_adaptive_isotonic"].predict(
                probabilities
            )
        )

    probability_columns = [
        column for column in output.columns
        if column not in {"TransactionID", "TransactionDT"}
    ]

    if len(probability_columns) != 14:
        raise ValueError(
            f"Expected 14 strategies, got {len(probability_columns)}."
        )

    for column in probability_columns:
        values = output[column]

        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values.")

        if not values.between(0, 1).all():
            raise ValueError(f"{column} contains invalid probabilities.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output.to_parquet(OUTPUT_PATH, index=False)

    metadata = {
        "experiment": "final_chronological_holdout",
        "prediction_stage": "label_blind",
        "test_target_loaded": False,
        "test_target_used": False,
        "test_rows": len(output),
        "input_feature_count": len(expected_features),
        "model_artifacts": {
            name: str(path.relative_to(ROOT))
            for name, path in MODEL_PATHS.items()
        },
        "calibration_files": {
            name: str((CALIBRATION_DIR / filename).relative_to(ROOT))
            for name, filename in CALIBRATION_FILES.items()
        },
        "strategies": probability_columns,
        "prediction_file": str(OUTPUT_PATH.relative_to(ROOT)),
        "calibration_protocol": (
            "Frozen static calibrators use initial frozen-model "
            "calibration predictions. Frozen adaptive calibrators "
            "use adaptation frozen-model predictions. Retrained, "
            "recent-only, and matched-size adaptive calibrators "
            "use their respective adaptation-calibration predictions."
        ),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nFINAL HOLDOUT PREDICTIONS SAVED")
    print(f"Rows: {len(output):,}")
    print(f"Strategies: {len(probability_columns)}")
    print(f"Predictions: {OUTPUT_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("\nNo final holdout labels were loaded or evaluated.")


if __name__ == "__main__":
    main()
