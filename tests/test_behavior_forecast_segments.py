import numpy as np
import pytest

from behavior_forecast.segments import (
    segment_by_gap,
    largest_segment,
    blocked_folds,
    forward_chaining_folds,
    valid_centers,
    seconds_to_frames,
)


def test_segment_by_gap_no_gap():
    ts = np.arange(0, 100) * 0.6
    segs = segment_by_gap(ts)
    assert segs == [(0, 100)]


def test_segment_by_gap_single_gap_splits_in_two():
    ts = np.concatenate([np.arange(0, 50) * 0.6, 50 * 0.6 + 300 + np.arange(0, 50) * 0.6])
    segs = segment_by_gap(ts)
    assert len(segs) == 2
    assert segs[0] == (0, 50)
    assert segs[1] == (50, 100)


def test_largest_segment_picks_bigger_half():
    ts = np.concatenate([np.arange(0, 30) * 0.6, 30 * 0.6 + 300 + np.arange(0, 70) * 0.6])
    start, stop = largest_segment(ts)
    assert (start, stop) == (30, 100)


def test_seconds_to_frames_rounds():
    assert seconds_to_frames(5.0, 0.6) == round(5.0 / 0.6)
    assert seconds_to_frames(0.1, 0.6) == 1  # never zero


def test_blocked_folds_cover_all_indices_exactly_once_as_test():
    n = 100
    folds = blocked_folds(n, n_blocks=5, purge_frames=5)
    covered = np.zeros(n, dtype=int)
    for f in folds:
        covered[f["test_start"]:f["test_stop"]] += 1
    assert (covered == 1).all()


def test_blocked_folds_purge_excludes_margin_from_train():
    n = 100
    purge = 10
    folds = blocked_folds(n, n_blocks=5, purge_frames=purge)
    for f in folds:
        lo = max(0, f["test_start"] - purge)
        hi = min(n, f["test_stop"] + purge)
        assert not f["train_mask"][lo:hi].any()
        # test block itself must also be excluded from train
        assert not f["train_mask"][f["test_start"]:f["test_stop"]].any()


def test_forward_chaining_folds_never_train_on_future():
    n = 100
    folds = forward_chaining_folds(n, n_blocks=5, purge_frames=5)
    for f in folds:
        train_idx = np.where(f["train_mask"])[0]
        assert train_idx.max() < f["test_start"]


def test_forward_chaining_skips_first_block():
    n = 100
    folds = forward_chaining_folds(n, n_blocks=5, purge_frames=5)
    assert all(f["test_start"] > 0 for f in folds)
    assert len(folds) == 4


def test_valid_centers_respects_history_and_horizon_bounds():
    n = 50
    centers = valid_centers(n, history_frames=10, horizon_frames=5)
    assert centers.min() >= 10
    assert centers.max() <= n - 5 - 1


def test_valid_centers_excludes_any_window_touching_masked_frames():
    n = 50
    mask = np.ones(n, dtype=bool)
    mask[20] = False  # one masked frame inside what would be a valid window's history
    centers = valid_centers(n, history_frames=10, horizon_frames=5, mask=mask)
    # any t whose [t-10, t] window includes index 20, or whose target t+5 == 20, must be excluded
    for t in centers:
        window_touches = (t - 10 <= 20 <= t) or (t + 5 == 20)
        assert not window_touches


def test_valid_centers_empty_when_recording_too_short():
    centers = valid_centers(n=10, history_frames=10, horizon_frames=5)
    assert len(centers) == 0
