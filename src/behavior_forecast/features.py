"""Feature construction for the behavior-forecasting pipeline.

Neural features are population-level only (per-recording summary
statistics / PCA components), never tied to a fixed neuron index, because
ROI index 7 in one recording is not known to be the same neuron as ROI
index 7 in another recording (no verified identity mapping is available).
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def behavior_history_features(velocity: np.ndarray, angular_velocity: np.ndarray,
                               reversal_vec: np.ndarray, t: int, history_frames: int) -> np.ndarray:
    """Summary of the trailing history window ending at (and including) t."""
    lo = t - history_frames
    v = velocity[lo:t + 1]
    av = angular_velocity[lo:t + 1]
    rv = reversal_vec[lo:t + 1]
    return np.array([
        v[-1], np.mean(v), np.std(v), v[-1] - v[0],
        av[-1], np.mean(av), np.std(av),
        rv[-1], np.mean(rv),
    ])


BEHAVIOR_HISTORY_FEATURE_NAMES = [
    "velocity_now", "velocity_mean", "velocity_std", "velocity_delta",
    "angvel_now", "angvel_mean", "angvel_std",
    "reversal_now", "reversal_rate",
]


def fit_neural_pca(train_trace: np.ndarray, n_components: int = 5) -> PCA:
    """Fit a PCA on population activity (timepoints x neurons) from the
    training segment only. n_components is capped by both dimensions."""
    k = max(1, min(n_components, train_trace.shape[1]))
    if k > train_trace.shape[0] - 1:
        raise ValueError(
            f"training segment too short ({train_trace.shape[0]} frames) for "
            f"{k}-component PCA; caller must skip this fold"
        )
    pca = PCA(n_components=k)
    pca.fit(train_trace)
    return pca


def neural_population_features(trace: np.ndarray, pca: PCA, t: int, history_frames: int) -> np.ndarray:
    """Population-level neural summary over the trailing history window:
    mean/std of overall activity plus the mean PCA-component scores. Never
    uses a fixed neuron's identity -- only aggregate statistics and
    directions of co-variation fit per recording.
    """
    lo = t - history_frames
    window = trace[lo:t + 1]
    mean_activity = window.mean(axis=1)
    pcs = pca.transform(window)
    feats = [mean_activity[-1], mean_activity.mean(), mean_activity.std()]
    feats.extend(pcs.mean(axis=0).tolist())
    feats.extend(pcs[-1].tolist())
    return np.array(feats)


def neural_population_feature_names(n_components: int) -> list[str]:
    names = ["activity_now", "activity_mean", "activity_std"]
    names.extend([f"pc{i}_mean" for i in range(n_components)])
    names.extend([f"pc{i}_now" for i in range(n_components)])
    return names
