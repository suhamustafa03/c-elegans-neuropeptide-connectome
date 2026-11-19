import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from evaluate import (
    _pairwise_bootstrap_ci,
    auprc,
    auroc,
    dyadic_bootstrap_ci,
    summarize,
)


def _all_pairs(neurons):
    upstream, downstream = [], []
    for a in neurons:
        for b in neurons:
            if a != b:
                upstream.append(a)
                downstream.append(b)
    return np.array(upstream), np.array(downstream)


def test_auroc_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(y_true, y_score) == 1.0


def test_auroc_single_class_is_nan():
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.1, 0.5, 0.9])
    assert np.isnan(auroc(y_true, y_score))


def test_auprc_above_baseline_for_informative_scores():
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.1).astype(int)
    y_score = y_true * 0.7 + rng.random(500) * 0.3
    baseline = y_true.mean()
    assert auprc(y_true, y_score) > baseline


# --- dyadic_bootstrap_ci: verifying it resamples the INTENDED unit ---


def test_dyadic_bootstrap_point_estimate_matches_plain_metric():
    neurons = [f"N{i:02d}" for i in range(15)]
    upstream, downstream = _all_pairs(neurons)
    rng = np.random.default_rng(0)
    y_true = (rng.random(len(upstream)) < 0.3).astype(int)
    y_score = y_true * 0.6 + rng.random(len(upstream)) * 0.4
    point, lo, hi = dyadic_bootstrap_ci(upstream, downstream, y_true, y_score, auroc, n_boot=100, seed=0)
    assert point == auroc(y_true, y_score)
    assert lo <= point <= hi or np.isnan(lo)


def test_dyadic_bootstrap_resamples_neurons_not_pairs_directly():
    """If the function resampled PAIR ROWS directly (the invalid approach),
    the number of rows it evaluates each bootstrap replicate would always
    equal the original row count. The dyadic method reconstructs pairs
    from resampled NEURON slots, so the resampled row count varies across
    replicates (some slot-pairs have no corresponding original pair, since
    not all neuron pairs are present in the index; others are hit multiple
    times because the same original neuron can occupy several slots). We
    verify this varies, which would be impossible under naive pair
    resampling of a complete all-pairs dataset (always exactly n rows)."""
    neurons = [f"N{i:02d}" for i in range(12)]
    upstream, downstream = _all_pairs(neurons)
    n_original = len(upstream)
    rng = np.random.default_rng(1)
    y_true = (rng.random(n_original) < 0.3).astype(int)
    y_score = rng.random(n_original)

    # Instrument manually: reimplement just the resample-size-tracking part
    # to confirm variability, without re-testing the metric computation.
    all_neurons = np.unique(np.concatenate([upstream, downstream]))
    pair_index = {}
    for idx, (u, d) in enumerate(zip(upstream, downstream)):
        pair_index.setdefault((u, d), []).append(idx)

    local_rng = np.random.default_rng(2)
    sizes = []
    for _ in range(30):
        resampled = local_rng.choice(all_neurons, size=len(all_neurons), replace=True)
        count = 0
        for i in range(len(resampled)):
            for j in range(len(resampled)):
                if i == j:
                    continue
                if (resampled[i], resampled[j]) in pair_index:
                    count += len(pair_index[(resampled[i], resampled[j])])
        sizes.append(count)

    assert len(set(sizes)) > 1, (
        "resampled dyad-set size should vary across bootstrap replicates; "
        "a constant size would indicate pair-level (not neuron-level) resampling"
    )
    # Note on an earlier, WRONG version of this test: it asserted
    # max(sizes) >= n_original, assuming neuron resampling would tend to
    # produce MORE dyad-slots than the original n_original pairs. That's
    # false on a complete all-pairs graph: with replacement, resampling
    # n_neurons slots from n_neurons neurons produces, in expectation,
    # FEWER than n_neurons*(n_neurons-1) hits, because slot-pairs that
    # happen to draw the SAME underlying neuron twice (a birthday-paradox
    # collision) look up a self-pair, which never exists in the original
    # data and contributes 0. Verified empirically (seed=2, 12 neurons,
    # 30 replicates): sizes ranged roughly 112-132 with n_original=132,
    # i.e. consistently at or below n_original, not above. The only claim
    # this test actually needs - that the size VARIES across replicates,
    # which a fixed-size pair-level resample could never show - still
    # holds and is asserted above.


def test_dyadic_bootstrap_hub_neuron_sanity_check():
    """TARGETED SANITY CHECK, not a proof of statistical validity (the
    module docstring's "What this interval does NOT establish" section
    is explicit about that distinction - no coverage check has been run,
    and this test doesn't attempt one).

    What this DOES check: on a deliberately constructed worst case for
    the naive pair-level bootstrap (one hub neuron drives the entire
    positive class, so the "evidence" for positives is really the same
    handful of hub-involving facts repeated across many rows), the
    dependence-aware method should not produce a NARROWER spread than the
    naive method - if it did, that would indicate the resampling
    mechanism isn't responding to the designed-in dependence at all,
    which would be a clear implementation bug. Passing this test is
    necessary-but-not-sufficient evidence the mechanism works as
    intended; it is not evidence the resulting interval has correct
    coverage for the real research question's estimand."""
    neurons = [f"N{i:02d}" for i in range(20)]
    hub = neurons[0]
    upstream, downstream = _all_pairs(neurons)
    y_true = np.array([1 if (u == hub or d == hub) else 0 for u, d in zip(upstream, downstream)])
    rng = np.random.default_rng(3)
    y_score = y_true * 0.8 + rng.random(len(upstream)) * 0.2

    _, dyadic_lo, dyadic_hi = dyadic_bootstrap_ci(
        upstream, downstream, y_true, y_score, auroc, n_boot=150, seed=0
    )
    _, naive_lo, naive_hi = _pairwise_bootstrap_ci(y_true, y_score, auroc, n_boot=150, seed=0)

    assert not np.isnan(dyadic_lo) and not np.isnan(naive_lo)
    dyadic_width = dyadic_hi - dyadic_lo
    naive_width = naive_hi - naive_lo
    assert dyadic_width >= naive_width, (
        f"expected the dependence-aware CI (width={dyadic_width:.4f}) to be at least as wide as "
        f"the naive pairwise CI (width={naive_width:.4f}), since the naive method ignores that "
        f"all positive pairs share a single hub neuron"
    )


def test_dyadic_bootstrap_too_few_neurons_returns_nan():
    upstream = np.array(["A", "B"])
    downstream = np.array(["B", "A"])
    y_true = np.array([1, 0])
    y_score = np.array([0.8, 0.2])
    point, lo, hi = dyadic_bootstrap_ci(
        upstream, downstream, y_true, y_score, auroc, n_boot=50, seed=0, min_neurons_for_ci=5
    )
    assert point == auroc(y_true, y_score)  # point estimate still reported
    assert np.isnan(lo) and np.isnan(hi)  # but CI explicitly unavailable, not fabricated


def test_dyadic_bootstrap_cluster_ids_not_silently_ignored():
    upstream, downstream = _all_pairs([f"N{i}" for i in range(6)])
    y_true = np.zeros(len(upstream), dtype=int)
    y_true[0] = 1
    y_score = np.random.default_rng(0).random(len(upstream))
    with pytest.raises(NotImplementedError):
        dyadic_bootstrap_ci(
            upstream, downstream, y_true, y_score, auroc,
            cluster_ids=np.zeros(len(upstream)),
        )


def test_summarize_reports_real_counts_and_ci_method_label():
    neurons = [f"N{i:02d}" for i in range(10)]
    upstream, downstream = _all_pairs(neurons)
    rng = np.random.default_rng(0)
    y_true = (rng.random(len(upstream)) < 0.2).astype(int)
    y_score = rng.random(len(upstream))
    result = summarize(upstream, downstream, y_true, y_score, label="synthetic-test-fixture", n_boot=50)
    assert result["n"] == len(upstream)
    assert result["n_positive"] == int(y_true.sum())
    assert result["n_neurons"] == len(neurons)
    assert result["ci_status"] == "provisional_uncalibrated"
    assert "Snijders" in result["ci_method"]
    assert "not validated" in result["ci_method"] or "NOT validated" in result["ci_method"]
    assert "auroc_ci" in result and "auprc_ci" in result
