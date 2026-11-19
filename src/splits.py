"""
Leakage-aware cross-validation for neuron-PAIR data.

Each observation is an (upstream, downstream) neuron pair. A single
neuron appears in many pairs (as both an upstream and downstream member),
so a random pair-level split lets the same neuron appear in both train and
test, letting a model partly "memorize" that neuron's typical behavior
rather than generalize. This module implements neuron-GROUPED splitting:
whole neurons (not just whole pairs) are held out, so a held-out fold
contains no pair whose upstream OR downstream neuron also appears in any
training pair.

This is stricter than a standard GroupKFold over a single ID column,
because each pair has TWO neuron identities that both need to be
"foreign" to the training set, not just one.
"""
from __future__ import annotations

import numpy as np


def neuron_grouped_folds(
    upstream: np.ndarray,
    downstream: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split pair-level data into n_splits folds, grouped by neuron so that
    no neuron appearing in a test fold (as upstream or downstream) also
    appears in that fold's training set.

    Parameters
    ----------
    upstream, downstream : arrays of neuron IDs, one per pair (same length)
    n_splits : number of folds
    seed : RNG seed for shuffling the neuron order before assigning folds

    Returns
    -------
    List of (train_idx, test_idx) index arrays into the original pair
    arrays, one tuple per fold. A given pair is dropped from a fold's
    train set if EITHER its upstream or downstream neuron is held out in
    that fold; it is only placed in the test set if BOTH its neurons are
    assigned to that fold's held-out group. Pairs that straddle a held-out
    neuron and a still-training neuron are dropped from that fold
    entirely (neither trained nor tested on), since they are neither a
    clean train example nor a clean held-out example for that fold - this
    is a deliberate, documented trade-off, not a bug: it costs some data
    per fold in exchange for a strictly leakage-free test set.
    """
    if len(upstream) != len(downstream):
        raise ValueError("upstream and downstream must be the same length")

    all_neurons = np.unique(np.concatenate([upstream, downstream]))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_neurons)
    neuron_fold = {n: i % n_splits for i, n in enumerate(shuffled)}

    upstream_fold = np.array([neuron_fold[n] for n in upstream])
    downstream_fold = np.array([neuron_fold[n] for n in downstream])

    folds = []
    for k in range(n_splits):
        is_test = (upstream_fold == k) & (downstream_fold == k)
        touches_k = (upstream_fold == k) | (downstream_fold == k)
        is_train = ~touches_k
        test_idx = np.where(is_test)[0]
        train_idx = np.where(is_train)[0]
        folds.append((train_idx, test_idx))
    return folds


def stimulated_neuron_folds(
    upstream: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split pair-level data by holding out whole UPSTREAM ("stimulated")
    neurons only, leaving downstream neurons free to repeat between train
    and test.

    Scientific justification (not a silent weakening of the design): in
    Randi et al.'s experiment, a neuron is optogenetically stimulated
    (upstream) while many downstream neurons are observed simultaneously
    via whole-brain imaging. Only a subset of all neurons were ever used
    as a stimulation target at all (stimulation requires a usable
    optogenetic line and optical access for that specific neuron), so a
    scientifically meaningful generalization question is: "does a model
    trained on connectivity patterns from the neurons we *could* stimulate
    generalize to predict the propagation pattern of a neuron we've never
    stimulated?" This is a real, asymmetric limitation of the experimental
    design, not an artificial relaxation invented to inflate fold sizes.

    This is a materially weaker leakage guarantee than
    `neuron_grouped_folds`: a downstream neuron CAN appear in both a
    fold's train and test pairs (just paired with a different, held-out
    upstream neuron each time), so a model could still partly exploit
    "neuron X is generally easy/hard to activate" patterns learned from
    training pairs where X was downstream. Use `neuron_grouped_folds` as
    the primary, strictest evaluation; use this as a secondary,
    explicitly-labeled comparison only when the strict split leaves too
    little data to be informative - never substitute this silently for
    the stricter split in reported headline results.
    """
    all_upstream = np.unique(upstream)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_upstream)
    neuron_fold = {n: i % n_splits for i, n in enumerate(shuffled)}
    fold_of = np.array([neuron_fold[n] for n in upstream])

    folds = []
    for k in range(n_splits):
        test_idx = np.where(fold_of == k)[0]
        train_idx = np.where(fold_of != k)[0]
        folds.append((train_idx, test_idx))
    return folds


def assert_no_neuron_leakage(
    upstream: np.ndarray,
    downstream: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Raise AssertionError if any neuron in the test fold also appears in
    the train fold. Intended for use in tests and as a runtime sanity
    check before trusting a held-out evaluation."""
    train_neurons = set(upstream[train_idx]) | set(downstream[train_idx])
    test_neurons = set(upstream[test_idx]) | set(downstream[test_idx])
    overlap = train_neurons & test_neurons
    assert not overlap, f"Neuron leakage between train and test: {sorted(overlap)[:10]}"
