
import numpy as np
import pytest

from driftsentinel.calibration.calibrator import ProbabilityCalibrator


@pytest.fixture
def calibration_data():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.4, 0.9])
    return y_true, probabilities


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_fit_and_predict(method, calibration_data):
    y_true, probabilities = calibration_data

    calibrator = ProbabilityCalibrator(method)
    calibrator.fit(y_true, probabilities)

    calibrated = calibrator.predict(probabilities)

    assert len(calibrated) == len(probabilities)
    assert np.isfinite(calibrated).all()
    assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_prediction_before_fit_raises_error():
    calibrator = ProbabilityCalibrator("sigmoid")

    with pytest.raises(RuntimeError):
        calibrator.predict([0.2, 0.8])


def test_invalid_method_raises_error():
    with pytest.raises(ValueError):
        ProbabilityCalibrator("temperature")


def test_invalid_probabilities_raise_error():
    calibrator = ProbabilityCalibrator("sigmoid")

    with pytest.raises(ValueError):
        calibrator.fit([0, 1], [0.2, 1.5])


def test_single_class_calibration_data_raises_error():
    calibrator = ProbabilityCalibrator("sigmoid")

    with pytest.raises(ValueError):
        calibrator.fit([0, 0, 0], [0.1, 0.2, 0.3])
