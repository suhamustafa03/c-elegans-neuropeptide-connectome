import numpy as np

from behavior_forecast.models import (
    fit_predict_regression,
    fit_predict_classification,
    persistence_velocity,
    persistence_reversal,
)


def test_fit_predict_regression_recovers_near_linear_signal():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 2))
    y_train = 3.0 * X_train[:, 0] - 1.0 * X_train[:, 1]
    X_test = rng.normal(size=(50, 2))
    y_test = 3.0 * X_test[:, 0] - 1.0 * X_test[:, 1]
    pred = fit_predict_regression(X_train, y_train, X_test)
    assert len(pred) == len(y_test)
    mae = np.mean(np.abs(pred - y_test))
    assert mae < 0.5  # should fit a near-noiseless linear signal closely


def test_fit_predict_regression_output_length_matches_input():
    X_train = np.ones((10, 3))
    y_train = np.arange(10.0)
    X_test = np.ones((4, 3))
    pred = fit_predict_regression(X_train, y_train, X_test)
    assert len(pred) == 4


def test_fit_predict_classification_returns_valid_probabilities():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(200, 2))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(50, 2))
    probs = fit_predict_classification(X_train, y_train, X_test)
    assert len(probs) == 50
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_fit_predict_classification_separates_clearly_informative_signal():
    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(300, 2))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = np.array([[5.0, 0.0], [-5.0, 0.0]])
    probs = fit_predict_classification(X_train, y_train, X_test)
    assert probs[0] > 0.9
    assert probs[1] < 0.1


def test_fit_predict_classification_falls_back_to_constant_for_single_class_train():
    X_train = np.ones((20, 2))
    y_train = np.zeros(20, dtype=int)  # only one class present
    X_test = np.ones((5, 2))
    probs = fit_predict_classification(X_train, y_train, X_test)
    assert np.all(probs == 0.0)


def test_fit_predict_classification_single_class_fallback_equals_prevalence():
    X_train = np.ones((10, 2))
    y_train = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # not actually single-class
    # Force single-class by using an all-ones label instead:
    y_train_single = np.ones(10, dtype=int)
    X_test = np.ones((3, 2))
    probs = fit_predict_classification(X_train, y_train_single, X_test)
    assert np.all(probs == 1.0)


def test_persistence_velocity_returns_value_at_center():
    velocity = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    centers = np.array([1, 3])
    pred = persistence_velocity(velocity, centers)
    np.testing.assert_array_equal(pred, [1.0, 3.0])


def test_persistence_reversal_returns_current_reversal_state():
    reversal_vec = np.array([0, 1, 0, 1, 1])
    centers = np.array([0, 1, 4])
    pred = persistence_reversal(reversal_vec, centers)
    np.testing.assert_array_equal(pred, [0.0, 1.0, 1.0])
