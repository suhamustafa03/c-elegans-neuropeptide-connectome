"""Time-gap segmentation and leakage-safe windowing for the behavior-forecasting
pipeline.

Recordings are not sampled on a perfectly uniform time grid: some contain a
single large real-time discontinuity (a pause), which would silently bridge
if windows were built from frame-index counts rather than actual timestamps.
"""
from __future__ import annotations

import numpy as np


def segment_by_gap(timestamps: np.ndarray, gap_factor: float = 3.0) -> list[tuple[int, int]]:
    """Split a recording into contiguous segments at any inter-frame gap
    larger than ``gap_factor`` times the recording's own median interval.

    Returns a list of (start, stop) index pairs (stop exclusive), covering
    the full recording with no overlaps.
    """
    n = len(timestamps)
    if n < 2:
        return [(0, n)]
    diffs = np.diff(timestamps)
    median_dt = np.median(diffs)
    gap_positions = np.where(diffs > gap_factor * median_dt)[0]
    bounds = [0] + [int(i) + 1 for i in gap_positions] + [n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def largest_segment(timestamps: np.ndarray, gap_factor: float = 3.0) -> tuple[int, int]:
    """Return the (start, stop) bounds of the longest contiguous segment."""
    segs = segment_by_gap(timestamps, gap_factor=gap_factor)
    return max(segs, key=lambda s: s[1] - s[0])


def median_dt(timestamps: np.ndarray) -> float:
    return float(np.median(np.diff(timestamps)))


def seconds_to_frames(seconds: float, dt: float) -> int:
    """Round a duration in seconds to a whole number of frames at interval dt."""
    return max(1, int(round(seconds / dt)))


def blocked_folds(n: int, n_blocks: int, purge_frames: int) -> list[dict]:
    """Build contiguous time-blocked train/test folds over indices [0, n).

    Each fold holds out one contiguous block as test; everything outside a
    ``purge_frames``-wide margin on both sides of that block is train. This
    is "blocked out-of-sample prediction": later blocks may appear in the
    training set of an earlier fold's test block. It is NOT a prospective
    forecast guarantee on its own -- see `forward_chaining_folds` for that.

    Returns a list of dicts with keys: test_start, test_stop (the held-out
    block, exclusive), train_mask (boolean array over [0, n)).
    """
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    folds = []
    for b in range(n_blocks):
        test_start, test_stop = edges[b], edges[b + 1]
        train_mask = np.ones(n, dtype=bool)
        lo = max(0, test_start - purge_frames)
        hi = min(n, test_stop + purge_frames)
        train_mask[lo:hi] = False
        folds.append({"test_start": int(test_start), "test_stop": int(test_stop), "train_mask": train_mask})
    return folds


def forward_chaining_folds(n: int, n_blocks: int, purge_frames: int) -> list[dict]:
    """Prospective variant: fold b's training set is restricted to indices
    strictly before its test block (minus the purge margin), i.e. the model
    never sees any future data relative to the test block it is scored on.
    The first block is skipped (no prior training data).
    """
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    folds = []
    for b in range(1, n_blocks):
        test_start, test_stop = edges[b], edges[b + 1]
        train_mask = np.zeros(n, dtype=bool)
        hi = max(0, test_start - purge_frames)
        train_mask[:hi] = True
        if train_mask.sum() == 0:
            continue
        folds.append({"test_start": int(test_start), "test_stop": int(test_stop), "train_mask": train_mask})
    return folds


def valid_centers(n: int, history_frames: int, horizon_frames: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Indices t such that the trailing history window [t-history_frames, t]
    and the forecast target t+horizon_frames both exist within [0, n), and
    (if a mask is given) every frame used -- history window, t itself, and
    the target frame -- is marked usable in `mask`.
    """
    lo = history_frames
    hi = n - horizon_frames
    if hi <= lo:
        return np.array([], dtype=int)
    candidates = np.arange(lo, hi)
    if mask is None:
        return candidates
    ok = np.array([
        mask[t - history_frames:t + 1].all() and mask[t + horizon_frames]
        for t in candidates
    ])
    return candidates[ok]
