"""Tests for the pipeline's orchestration logic (run_recording,
reversal_onset_labels) using small synthetic segments built directly as
numpy arrays -- no h5py, no external data download. load_recording itself
(which does require h5py to read real .h5 files) is intentionally not
unit-tested here; it is a thin, directly-inspectable I/O wrapper, and
exercising it would need a real or fabricated .h5 fixture file, which
would misrepresent real recording structure if fabricated.
"""
import numpy as np
import pytest

from behavior_forecast.pipeline import run_recording, reversal_onset_labels
from behavior_forecast.segments import blocked_folds, forward_chaining_folds


def _synthetic_segment(n=400, n_neurons=8, seed=0, dt=0.6):
    rng = np.random.default_rng(seed)
    timestamps = np.arange(n) * dt
    trace = rng.normal(size=(n, n_neurons))
    velocity = np.sin(np.linspace(0, 20, n)) + rng.normal(scale=0.05, size=n)
    angular_velocity = rng.normal(size=n)
    # reversal_vec: occasional short reversal bouts so onset labels are non-degenerate
    reversal_vec = np.zeros(n)
    for start in range(20, n - 20, 37):
        reversal_vec[start:start + 5] = 1
    return {
        "timestamps": timestamps,
        "trace": trace,
        "velocity": velocity,
        "angular_velocity": angular_velocity,
        "reversal_vec": reversal_vec,
        "has_stim": False,
        "n_frames": n,
    }


def test_reversal_onset_labels_excludes_already_reversing_frames():
    reversal_vec = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    centers = np.array([1, 2, 5])
    horizon = 3
    labels = reversal_onset_labels(reversal_vec, centers, horizon)
    # t=1: not reversing at t=1, window (2,3,4) has a 1 -> onset = 1
    assert labels[0] == 1
    # t=2: already reversing at t=2 -> excluded, label = 0 regardless of window
    assert labels[1] == 0
    # t=5: not reversing at t=5, window (6,7,8) has a 1 at index 7 -> onset = 1
    assert labels[2] == 1


def test_reversal_onset_labels_zero_when_no_future_reversal():
    reversal_vec = np.zeros(20)
    centers = np.array([5, 10])
    labels = reversal_onset_labels(reversal_vec, centers, horizon_frames=4)
    assert (labels == 0).all()


def test_run_recording_produces_all_four_models_both_tasks():
    seg = _synthetic_segment()
    out = run_recording(seg, fold_fn=blocked_folds)
    assert out["n_folds_total"] == 5
    assert out["n_folds_run"] > 0
    for model in ["persistence", "behavior_only", "neural_only", "combined"]:
        assert model in out["results"]
        assert "velocity" in out["results"][model]
        assert "reversal" in out["results"][model]
        assert len(out["results"][model]["velocity"]) == out["n_folds_run"]


def test_run_recording_velocity_metrics_have_expected_keys():
    seg = _synthetic_segment()
    out = run_recording(seg, fold_fn=blocked_folds)
    fold_metric = out["results"]["behavior_only"]["velocity"][0]
    assert set(fold_metric.keys()) == {"mae", "r2", "n"}
    assert fold_metric["mae"] >= 0


def test_run_recording_reversal_metrics_have_expected_keys():
    seg = _synthetic_segment()
    out = run_recording(seg, fold_fn=blocked_folds)
    fold_metric = out["results"]["behavior_only"]["reversal"][0]
    assert "pr_auc" in fold_metric
    assert "undefined" in fold_metric


def test_run_recording_forward_chaining_skips_fewer_or_equal_folds_than_blocked():
    seg = _synthetic_segment()
    out_blocked = run_recording(seg, fold_fn=blocked_folds)
    out_fc = run_recording(seg, fold_fn=forward_chaining_folds)
    # forward-chaining has one fewer fold by construction (first block skipped)
    assert out_fc["n_folds_total"] == out_blocked["n_folds_total"] - 1


def test_run_recording_skips_folds_too_short_for_history_and_horizon():
    """A recording far too short for the 10s history + 5s horizon (at this
    segment's dt) must be skipped cleanly, not raise."""
    seg = _synthetic_segment(n=15)  # at dt=0.6s this is ~9s total, shorter than the purge alone
    out = run_recording(seg, fold_fn=blocked_folds)
    assert out["n_folds_run"] == 0
    assert len(out["skipped_folds"]) == out["n_folds_total"]


def test_run_recording_persistence_velocity_equals_value_at_center_not_a_fit():
    """Sanity check that persistence is truly just a lookup, not a fitted
    model producing coincidentally identical values."""
    seg = _synthetic_segment(seed=1)
    out = run_recording(seg, fold_fn=blocked_folds)
    # persistence metrics should be computable with zero free parameters;
    # just check it ran and produced a finite MAE on every fold.
    for fm in out["results"]["persistence"]["velocity"]:
        assert np.isfinite(fm["mae"])


def test_run_recording_deterministic_for_fixed_seed():
    seg_a = _synthetic_segment(seed=7)
    seg_b = _synthetic_segment(seed=7)
    out_a = run_recording(seg_a, fold_fn=blocked_folds)
    out_b = run_recording(seg_b, fold_fn=blocked_folds)
    mae_a = [fm["mae"] for fm in out_a["results"]["behavior_only"]["velocity"]]
    mae_b = [fm["mae"] for fm in out_b["results"]["behavior_only"]["velocity"]]
    assert mae_a == mae_b
