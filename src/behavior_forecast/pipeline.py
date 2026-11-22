"""End-to-end behavior-forecasting pipeline for one recording.

Requires h5py to read the source .h5 files; h5py is intentionally NOT a
dependency of this project's own requirements.txt (the raw/derived
propagation-style data this reads lives outside Git in an external,
isolated environment -- see docs/DATA_LICENSES.md). This module is run
with that external interpreter, not the project's own .venv.
"""
from __future__ import annotations

import numpy as np

from .segments import largest_segment, median_dt, seconds_to_frames, blocked_folds, forward_chaining_folds, valid_centers
from .features import behavior_history_features, neural_population_features, fit_neural_pca
from .models import fit_predict_regression, fit_predict_classification, persistence_velocity, persistence_reversal
from .metrics import regression_metrics, classification_metrics

HISTORY_SECONDS = 10.0
HORIZON_SECONDS = 5.0
N_BLOCKS = 5
N_PCA_COMPONENTS = 5


def reversal_onset_labels(reversal_vec: np.ndarray, centers: np.ndarray, horizon_frames: int) -> np.ndarray:
    """1 if a reversal begins strictly within (t, t+horizon_frames] and the
    animal was not already reversing at t, else 0. Excluding "already
    reversing at t" from a positive label is deliberate: it defines
    "onset" as a new event, not merely "still reversing" -- see
    docs/BEHAVIOR_FORECAST_RESULTS.md for why persistence's ROC-AUC is
    below 0.5 as a direct consequence of this definition."""
    labels = np.zeros(len(centers), dtype=int)
    for i, t in enumerate(centers):
        window = reversal_vec[t + 1:t + horizon_frames + 1]
        was_reversing_at_t = reversal_vec[t] > 0
        labels[i] = int((not was_reversing_at_t) and (window > 0).any())
    return labels


def load_recording(h5_path, segment_choice: str = "largest"):
    """Load one recording's arrays. segment_choice: 'largest' (primary) or
    'all' (returns every gap-delimited segment for the sensitivity check)."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        ts = f["timing"]["timestamp_confocal"][:]
        trace = f["gcamp"]["trace_array"][:]
        velocity = f["behavior"]["velocity"][:]
        angular_velocity = f["behavior"]["angular_velocity"][:]
        reversal_vec = f["behavior"]["reversal_vec"][:]
        has_stim = "stim_begin_confocal" in f["timing"]

    from .segments import segment_by_gap
    segs = segment_by_gap(ts) if segment_choice == "all" else [largest_segment(ts)]
    out = []
    for start, stop in segs:
        out.append({
            "timestamps": ts[start:stop],
            "trace": trace[start:stop],
            "velocity": velocity[start:stop],
            "angular_velocity": angular_velocity[start:stop],
            "reversal_vec": reversal_vec[start:stop],
            "has_stim": has_stim,
            "n_frames": stop - start,
        })
    return out


def run_recording(segment: dict, fold_fn=blocked_folds,
                   history_seconds: float = HISTORY_SECONDS,
                   horizon_seconds: float = HORIZON_SECONDS) -> dict:
    """Run the full model comparison for one (segment of a) recording.
    Returns per-model metrics for velocity regression and reversal
    classification, aggregated across this recording's folds.

    `history_seconds`/`horizon_seconds` default to this project's reported
    protocol (10s/5s) but are exposed as parameters so a horizon/history
    sensitivity sweep can be run without editing this function -- see
    docs/BEHAVIOR_FORECAST_RESULTS.md for whether such a sweep was run on
    real data and why/why not.
    """
    ts = segment["timestamps"]
    trace = segment["trace"]
    velocity = segment["velocity"]
    angular_velocity = segment["angular_velocity"]
    reversal_vec = segment["reversal_vec"]
    n = segment["n_frames"]

    dt = median_dt(ts)
    history_frames = seconds_to_frames(history_seconds, dt)
    horizon_frames = seconds_to_frames(horizon_seconds, dt)
    purge_frames = history_frames + horizon_frames

    folds = fold_fn(n, N_BLOCKS, purge_frames)

    results = {m: {"velocity": [], "reversal": []} for m in
               ["persistence", "behavior_only", "neural_only", "combined"]}
    skipped_folds = []

    for fold in folds:
        train_mask = fold["train_mask"]
        test_start, test_stop = fold["test_start"], fold["test_stop"]

        train_centers = valid_centers(n, history_frames, horizon_frames, mask=train_mask)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_start:test_stop] = True
        test_centers = valid_centers(n, history_frames, horizon_frames, mask=test_mask)

        if len(train_centers) < max(20, 2 * horizon_frames) or len(test_centers) == 0:
            skipped_folds.append({"test_start": test_start, "test_stop": test_stop, "reason": "too_few_centers"})
            continue

        try:
            pca = fit_neural_pca(trace[train_mask], n_components=N_PCA_COMPONENTS)
        except ValueError:
            skipped_folds.append({"test_start": test_start, "test_stop": test_stop, "reason": "train_segment_too_short_for_pca"})
            continue

        def build_features(centers):
            beh = np.stack([behavior_history_features(velocity, angular_velocity, reversal_vec, t, history_frames) for t in centers])
            neu = np.stack([neural_population_features(trace, pca, t, history_frames) for t in centers])
            return beh, neu

        beh_train, neu_train = build_features(train_centers)
        beh_test, neu_test = build_features(test_centers)

        y_vel_train = velocity[train_centers + horizon_frames]
        y_vel_test = velocity[test_centers + horizon_frames]

        y_rev_train = reversal_onset_labels(reversal_vec, train_centers, horizon_frames)
        y_rev_test = reversal_onset_labels(reversal_vec, test_centers, horizon_frames)

        # --- velocity regression ---
        pred = {
            "persistence": persistence_velocity(velocity, test_centers),
            "behavior_only": fit_predict_regression(beh_train, y_vel_train, beh_test),
            "neural_only": fit_predict_regression(neu_train, y_vel_train, neu_test),
            "combined": fit_predict_regression(np.hstack([beh_train, neu_train]), y_vel_train, np.hstack([beh_test, neu_test])),
        }
        for m, p in pred.items():
            results[m]["velocity"].append(regression_metrics(y_vel_test, p))

        # --- reversal-onset classification ---
        pred_c = {
            "persistence": persistence_reversal(reversal_vec, test_centers),
            "behavior_only": fit_predict_classification(beh_train, y_rev_train, beh_test),
            "neural_only": fit_predict_classification(neu_train, y_rev_train, neu_test),
            "combined": fit_predict_classification(np.hstack([beh_train, neu_train]), y_rev_train, np.hstack([beh_test, neu_test])),
        }
        for m, p in pred_c.items():
            results[m]["reversal"].append(classification_metrics(y_rev_test, p))

    return {
        "n_folds_run": len(results["persistence"]["velocity"]),
        "n_folds_total": len(folds),
        "skipped_folds": skipped_folds,
        "dt": dt,
        "history_frames": history_frames,
        "horizon_frames": horizon_frames,
        "results": results,
    }
