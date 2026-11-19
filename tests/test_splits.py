import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from splits import assert_no_neuron_leakage, neuron_grouped_folds, stimulated_neuron_folds


def _make_all_pairs(neurons):
    upstream, downstream = [], []
    for a in neurons:
        for b in neurons:
            if a != b:
                upstream.append(a)
                downstream.append(b)
    return np.array(upstream), np.array(downstream)


def test_no_leakage_across_all_folds():
    neurons = [f"N{i:03d}" for i in range(40)]
    upstream, downstream = _make_all_pairs(neurons)
    folds = neuron_grouped_folds(upstream, downstream, n_splits=5, seed=0)
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert len(test_idx) > 0, "a fold with 40 neurons over 5 splits should have some fully-held-out pairs"
        assert_no_neuron_leakage(upstream, downstream, train_idx, test_idx)


def test_every_pair_appears_as_test_at_most_once_and_train_sets_shrink_vs_naive():
    neurons = [f"N{i:03d}" for i in range(20)]
    upstream, downstream = _make_all_pairs(neurons)
    folds = neuron_grouped_folds(upstream, downstream, n_splits=4, seed=1)
    test_counts = np.zeros(len(upstream), dtype=int)
    for _, test_idx in folds:
        test_counts[test_idx] += 1
    assert test_counts.max() <= 1

    # Because pairs straddling the test group are dropped from training
    # (not just test), each fold's train set should be meaningfully
    # smaller than "everything not in this fold's test set" would be
    # under a naive (leaky) pair-level split.
    for train_idx, test_idx in folds:
        naive_train_size = len(upstream) - len(test_idx)
        assert len(train_idx) < naive_train_size


def test_deterministic_with_seed():
    neurons = [f"N{i:03d}" for i in range(15)]
    upstream, downstream = _make_all_pairs(neurons)
    folds_a = neuron_grouped_folds(upstream, downstream, n_splits=3, seed=7)
    folds_b = neuron_grouped_folds(upstream, downstream, n_splits=3, seed=7)
    for (tr_a, te_a), (tr_b, te_b) in zip(folds_a, folds_b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(te_a, te_b)


def test_stimulated_neuron_folds_every_pair_tested_exactly_once():
    neurons = [f"N{i:03d}" for i in range(20)]
    upstream, downstream = _make_all_pairs(neurons)
    folds = stimulated_neuron_folds(upstream, n_splits=4, seed=0)
    test_counts = np.zeros(len(upstream), dtype=int)
    for _, test_idx in folds:
        test_counts[test_idx] += 1
    assert np.all(test_counts == 1), "every pair should be tested in exactly one fold"


def test_stimulated_neuron_folds_only_upstream_held_out():
    neurons = [f"N{i:03d}" for i in range(20)]
    upstream, downstream = _make_all_pairs(neurons)
    folds = stimulated_neuron_folds(upstream, n_splits=4, seed=0)
    for train_idx, test_idx in folds:
        train_up = set(upstream[train_idx])
        test_up = set(upstream[test_idx])
        assert not (train_up & test_up), "upstream (stimulated) neurons must not repeat between train/test"
        train_down = set(downstream[train_idx])
        test_down = set(downstream[test_idx])
        # Downstream neurons ARE expected to repeat - this is the documented,
        # deliberate trade-off versus neuron_grouped_folds.
        assert train_down & test_down, (
            "this test fixture (all-pairs over a shared neuron set) should show downstream "
            "overlap by construction; if this fails, stimulated_neuron_folds may have "
            "accidentally become as strict as neuron_grouped_folds"
        )


def test_leakage_assertion_catches_real_leak():
    upstream = np.array(["A", "B", "C"])
    downstream = np.array(["B", "C", "A"])
    train_idx = np.array([0, 1])  # neurons A, B, C all present via these two pairs
    test_idx = np.array([2])  # pair C->A: both C and A are in train_idx's neuron set
    try:
        assert_no_neuron_leakage(upstream, downstream, train_idx, test_idx)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "expected a leakage assertion to fire for an overlapping neuron set"
