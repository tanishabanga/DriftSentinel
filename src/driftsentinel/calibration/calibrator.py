
"""Probability-only calibration utilities."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Fit a calibration mapping without retraining the base model."""

    def __init__(self, method: str):
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError(
                "method must be either 'sigmoid' or 'isotonic'."
            )

        self.method = method
        self.model = None

    def fit(self, y_true, probabilities):
        y_true = np.asarray(y_true)
        probabilities = np.asarray(probabilities, dtype=float)

        self._validate(y_true, probabilities)

        if len(np.unique(y_true)) < 2:
            raise ValueError("Calibration data must contain both classes.")

        if self.method == "sigmoid":
            # Platt-style scaling on the log-odds of the probabilities.
            epsilon = 1e-7
            clipped = np.clip(probabilities, epsilon, 1 - epsilon)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)

            self.model = LogisticRegression(
                solver="lbfgs",
                random_state=42,
            )
            self.model.fit(logits, y_true)

        else:
            self.model = IsotonicRegression(
                y_min=0,
                y_max=1,
                out_of_bounds="clip",
            )
            self.model.fit(probabilities, y_true)

        return self

    def predict(self, probabilities):
        if self.model is None:
            raise RuntimeError("Calibrator must be fitted before prediction.")

        probabilities = np.asarray(probabilities, dtype=float)

        if probabilities.ndim != 1:
            raise ValueError("Probabilities must be one-dimensional.")

        if not np.isfinite(probabilities).all():
            raise ValueError("Probabilities must be finite.")

        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Probabilities must be between 0 and 1.")

        if self.method == "sigmoid":
            epsilon = 1e-7
            clipped = np.clip(probabilities, epsilon, 1 - epsilon)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            return self.model.predict_proba(logits)[:, 1]

        return self.model.predict(probabilities)

    @staticmethod
    def _validate(y_true, probabilities):
        if y_true.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("Inputs must be one-dimensional.")

        if len(y_true) == 0 or len(y_true) != len(probabilities):
            raise ValueError("Inputs must have equal, non-zero length.")

        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("Labels must contain only 0 and 1.")

        if not np.isfinite(probabilities).all():
            raise ValueError("Probabilities must be finite.")

        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Probabilities must be between 0 and 1.")
