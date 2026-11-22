import numpy as np
import pytest

from behavior_forecast.features import (
    behavior_history_features,
    BEHAVIOR_HISTORY_FEATURE_NAMES,
    fit_neural_pca,
    neural_population_features,
    neural_population_feature_names,
)


def test_behavior_history_features_length_matches_names():
    n = 50
    velocity = np.linspace(0, 1, n)
    angular_velocity = np.linspace(-1, 1, n)
    reversal_vec = np.zeros(n)
    feats = behavior_history_features(velocity, angular_velocity, reversal_vec, t=20, history_frames=10)
    assert len(feats) == len(BEHAVIOR_HISTORY_FEATURE_NAMES) == 9


def test_behavior_history_features_values_on_known_window():
    n = 30
    velocity = np.arange(n, dtype=float)  # 0,1,2,...
    angular_velocity = np.zeros(n)
    reversal_vec = np.zeros(n)
    reversal_vec[15] = 1  # currently reversing at t=20's window? no, check window [10,20]
    feats = behavior_history_features(velocity, angular_velocity, reversal_vec, t=20, history_frames=10)
    # window is velocity[10:21] = 10..20
    assert feats[0] == 20.0  # velocity_now
    assert feats[1] == pytest.approx(15.0)  # mean of 10..20
    assert feats[3] == pytest.approx(10.0)  # velocity_now - velocity[10] = 20-10
    assert feats[7] == 0.0  # reversal_now (reversal_vec[20] == 0)
    assert feats[8] == pytest.approx(1.0 / 11.0)  # reversal_rate: one 1 in an 11-frame window


def test_fit_neural_pca_caps_components_by_neuron_count():
    rng = np.random.default_rng(0)
    train_trace = rng.normal(size=(100, 3))  # only 3 neurons
    pca = fit_neural_pca(train_trace, n_components=5)
    assert pca.n_components_ == 3


def test_fit_neural_pca_raises_when_training_segment_too_short():
    rng = np.random.default_rng(0)
    train_trace = rng.normal(size=(3, 10))  # only 3 frames, 10 neurons, k=5
    with pytest.raises(ValueError):
        fit_neural_pca(train_trace, n_components=5)


def test_neural_population_features_length_matches_names():
    rng = np.random.default_rng(1)
    trace = rng.normal(size=(100, 10))
    pca = fit_neural_pca(trace, n_components=4)
    feats = neural_population_features(trace, pca, t=50, history_frames=10)
    names = neural_population_feature_names(4)
    assert len(feats) == len(names) == 3 + 4 + 4


def test_neural_population_features_deterministic_for_fixed_input():
    rng = np.random.default_rng(42)
    trace = rng.normal(size=(50, 6))
    pca = fit_neural_pca(trace, n_components=2)
    feats_a = neural_population_features(trace, pca, t=30, history_frames=5)
    feats_b = neural_population_features(trace, pca, t=30, history_frames=5)
    np.testing.assert_array_equal(feats_a, feats_b)
