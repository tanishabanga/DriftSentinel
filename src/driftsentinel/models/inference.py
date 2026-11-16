
"""Production inference service for DriftSentinel.

Loads the trained XGBoost model, validates the feature schema,
prepares transaction data, calibrates fraud probabilities,
and applies cost-sensitive decision policies.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from driftsentinel.models.exceptions import InferenceInputError
from driftsentinel.policies.costs import DecisionCosts
from driftsentinel.policies.decision import make_decisions

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "artifacts/xgboost/fraud_model.json"
)

DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT / "artifacts/xgboost/feature_schema.json"
)

DEFAULT_CALIBRATOR_PATH = (
    PROJECT_ROOT
    / "artifacts/xgboost/calibration/sigmoid_calibrator.joblib"
)


# ---------------------------------------------------------
# INFERENCE SERVICE
# ---------------------------------------------------------

class FraudInferenceService:
    """Load the trained model and perform validated fraud inference."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
        calibrator_path: Path = DEFAULT_CALIBRATOR_PATH,
        costs: DecisionCosts | None = None,
    ):

        self.model_path = Path(model_path)
        self.schema_path = Path(schema_path)
        self.calibrator_path = Path(calibrator_path)

        self.costs = costs if costs is not None else DecisionCosts()

        # Load and validate feature schema.
        self.schema = self._load_schema()
        self._validate_schema()

        # Extract schema metadata.
        self.feature_names = self.schema["feature_names"]
        self.numerical_columns = self.schema["numerical_columns"]
        self.categorical_columns = self.schema["categorical_columns"]

        self.categorical_vocabularies = (
            self.schema["categorical_vocabularies"]
        )

        # Load trained XGBoost model.
        self.model = self._load_model()

        # Load sigmoid calibrator.
        self.calibrator = self._load_calibrator()

    # -----------------------------------------------------
    # SCHEMA LOADING
    # -----------------------------------------------------

    def _load_schema(self):
        """Load feature schema from JSON."""

        if not self.schema_path.exists():
            raise FileNotFoundError(
                f"Feature schema not found: {self.schema_path}"
            )

        with open(self.schema_path, "r", encoding="utf-8") as file:
            schema = json.load(file)

        return schema

    def _validate_schema(self):
        """Validate schema structure and feature consistency."""

        required_keys = {
            "schema_version",
            "feature_count",
            "feature_names",
            "numerical_columns",
            "categorical_columns",
            "categorical_vocabularies",
            "missing_value_policy",
            "unknown_category_policy",
        }

        missing_keys = required_keys - set(self.schema.keys())

        if missing_keys:
            raise ValueError(
                f"Schema is missing keys: {sorted(missing_keys)}"
            )

        feature_names = self.schema["feature_names"]
        numerical = self.schema["numerical_columns"]
        categorical = self.schema["categorical_columns"]

        if len(feature_names) != self.schema["feature_count"]:
            raise ValueError(
                "Feature count does not match feature_names length."
            )

        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Duplicate feature names found in schema.")

        if set(numerical) & set(categorical):
            raise ValueError(
                "Numerical and categorical columns overlap."
            )

        if set(numerical) | set(categorical) != set(feature_names):
            raise ValueError(
                "Numerical and categorical columns do not cover "
                "the complete feature schema."
            )

        if set(self.schema["categorical_vocabularies"]) != set(categorical):
            raise ValueError(
                "Categorical vocabularies do not match categorical columns."
            )

        if self.schema["missing_value_policy"] != "preserve_missing_values":
            raise ValueError("Unsupported missing-value policy.")

        if self.schema["unknown_category_policy"] != "map_to_missing":
            raise ValueError("Unsupported unknown-category policy.")

    # -----------------------------------------------------
    # MODEL LOADING
    # -----------------------------------------------------

    def _load_model(self):
        """Load trained XGBoost model."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"XGBoost model not found: {self.model_path}"
            )

        model = XGBClassifier()
        model.load_model(str(self.model_path))

        return model

    # -----------------------------------------------------
    # CALIBRATOR LOADING
    # -----------------------------------------------------

    def _load_calibrator(self):
        """Load trained probability calibrator."""

        if not self.calibrator_path.exists():
            raise FileNotFoundError(
                f"Calibrator not found: {self.calibrator_path}"
            )

        calibrator = joblib.load(self.calibrator_path)

        if not (
            hasattr(calibrator, "predict_proba")
            or hasattr(calibrator, "predict")
        ):
            raise TypeError(
                "Calibrator must implement predict_proba() or predict()."
            )

        return calibrator

    # -----------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------

    def _validate_input(self, data):
        """Validate incoming transaction data."""

        if isinstance(data, pd.Series):
            data = data.to_frame().T

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame or Series."
            )

        if data.empty:
            raise InferenceInputError("Input transaction data cannot be empty.")

        if data.columns.duplicated().any():
            raise InferenceInputError("Duplicate input columns detected.")

        missing_columns = (
            set(self.feature_names) - set(data.columns)
        )

        if missing_columns:
            raise InferenceInputError(
                "Input is missing required features: "
                f"{sorted(missing_columns)}"
            )

        # Extra columns are ignored.
        return data.loc[:, self.feature_names].copy()

    # -----------------------------------------------------
    # FEATURE PREPARATION
    # -----------------------------------------------------

    def _prepare_features(self, data):
        """Prepare input data according to the saved schema."""

        data = self._validate_input(data)

        # Convert numerical features safely.
        for column in self.numerical_columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            ).astype("float32")

        # Convert categorical features using saved vocabularies.
        for column in self.categorical_columns:

            values = data[column].astype("string")

            vocabulary = self.categorical_vocabularies[column]

            # Unknown categories become missing values.
            values = values.where(
                values.isin(vocabulary),
                other=pd.NA
            )

            # Preserve the exact training category vocabulary.
            data[column] = pd.Categorical(
                values,
                categories=vocabulary
            )

        # Maintain exact feature order.
        data = data[self.feature_names]

        return data

    # -----------------------------------------------------
    # RAW MODEL PREDICTIONS
    # -----------------------------------------------------

    def predict_raw(self, data):
        """Generate uncalibrated fraud probabilities."""

        features = self._prepare_features(data)

        probabilities = self.model.predict_proba(features)

        probabilities = np.asarray(probabilities, dtype=float)

        if probabilities.ndim != 2:
            raise ValueError(
                "Unexpected probability output from XGBoost."
            )

        if probabilities.shape[1] != 2:
            raise ValueError(
                "Expected a binary fraud classification model."
            )

        # Identify the fraud class rather than assuming column order.
        classes = list(self.model.classes_)

        if 1 not in classes:
            raise ValueError(
                "The trained model does not contain fraud class 1."
            )

        fraud_index = classes.index(1)

        probabilities = probabilities[:, fraud_index]

        self._validate_probabilities(probabilities)

        return probabilities

    # -----------------------------------------------------
    # PROBABILITY VALIDATION
    # -----------------------------------------------------

    @staticmethod
    def _validate_probabilities(probabilities):
        """Validate probability values."""

        probabilities = np.asarray(probabilities, dtype=float)

        if probabilities.ndim != 1:
            raise ValueError("Probabilities must be one-dimensional.")

        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite probabilities detected.")

        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Probabilities must be between 0 and 1.")

    # -----------------------------------------------------
    # PROBABILITY CALIBRATION
    # -----------------------------------------------------

    def predict_calibrated(self, raw_probabilities):
        """Apply the saved sigmoid probability calibrator."""

        raw_probabilities = np.asarray(
            raw_probabilities,
            dtype=float
        )

        self._validate_probabilities(raw_probabilities)

        if hasattr(self.calibrator, "predict_proba"):

            calibrated = self.calibrator.predict_proba(
                raw_probabilities.reshape(-1, 1)
            )

            calibrated = np.asarray(calibrated, dtype=float)

            if calibrated.ndim != 2:
                raise ValueError(
                    "Unexpected calibrator probability output."
                )

            if calibrated.shape[1] != 2:
                raise ValueError(
                    "Expected a binary probability calibrator."
                )

            classes = getattr(
                self.calibrator,
                "classes_",
                np.array([0, 1])
            )

            classes = list(classes)

            if 1 not in classes:
                raise ValueError(
                    "Calibrator does not contain fraud class 1."
                )

            calibrated = calibrated[:, classes.index(1)]

        else:
            calibrated = self.calibrator.predict(raw_probabilities)

            calibrated = np.asarray(calibrated, dtype=float).reshape(-1)

        self._validate_probabilities(calibrated)

        if len(calibrated) != len(raw_probabilities):
            raise ValueError(
                "Calibrator changed the number of predictions."
            )

        return calibrated

    # -----------------------------------------------------
    # COST-SENSITIVE DECISIONS
    # -----------------------------------------------------

    def make_decisions(self, probabilities):
        """Apply the configured cost-sensitive decision policy."""

        return make_decisions(
            probabilities,
            costs=self.costs
        )

    # -----------------------------------------------------
    # COMPLETE INFERENCE PIPELINE
    # -----------------------------------------------------

    
    def predict(self, data):
        """Run fraud inference and preserve transaction metadata."""

        # Preserve metadata before feature validation removes extra columns.
        if isinstance(data, pd.DataFrame):
            output_index = data.index.copy()

            transaction_ids = (
                data["TransactionID"].to_numpy(copy=True)
                if "TransactionID" in data.columns
                else None
            )
        elif isinstance(data, pd.Series):
            output_index = pd.RangeIndex(1)
            transaction_ids = None
        else:
            raise TypeError(
                "Input must be a pandas DataFrame or Series."
            )

        # Step 1: Generate raw probabilities.
        raw_probabilities = self.predict_raw(data)

        # Step 2: Calibrate probabilities.
        calibrated_probabilities = self.predict_calibrated(
            raw_probabilities
        )

        # Step 3: Apply cost-sensitive decisions.
        decisions = self.make_decisions(
            calibrated_probabilities
        )

        # Step 4: Construct output with transaction metadata.
        results = pd.DataFrame(
            {
                "raw_probability": raw_probabilities,
                "calibrated_probability": calibrated_probabilities,
                "action": decisions["action"].to_numpy(),
                "expected_cost": decisions["expected_cost"].to_numpy(),
            },
            index=output_index
        )

        if transaction_ids is not None:
            results.insert(0, "TransactionID", transaction_ids)

        return results
