"""
Audit the cross-validation splitters on REAL neuron/pair structure.

Important honesty note: the actual propagation target data (Randi et al.)
has not been obtained yet (see docs/fetch_propagation_data.md), so this
audit cannot report the real positive/negative class balance for the
actual research question. Instead, it runs both split strategies over the
REAL 302-neuron set and REAL pair universe, using the already-downloaded
White et al. 1986 chemical-synapse connectivity as a STAND-IN binary
label purely to exercise the splitters on genuine network structure
(which has realistic hub-neuron degree imbalance, unlike a uniform random
label would) and to report real, non-fabricated fold statistics.

This is a dry run of the audit mechanism, not a result for the research
question. Re-run this script once the real propagation labels exist
(after docs/fetch_propagation_data.md is completed) to get the audit that
actually matters.

Usage:
    python src/audit_splits.py
"""
from __future__ import annotations

import pathlib

import numpy as np

from features import load_white_connectome
from neuron_id import normalize_neuron_id
from splits import neuron_grouped_folds, stimulated_neuron_folds, assert_no_neuron_leakage

ROOT = pathlib.Path(__file__).resolve().parent.parent
WHITE_PATH = ROOT / "data" / "raw" / "synaptic_connectome" / "aconnectome_white_1986_whole.csv"
NP_PATH = ROOT / "data" / "raw" / "neuropeptide_connectome" / "neuropeptide_connectome_short_range_model.csv"


def load_neuron_set() -> list[str]:
    import csv
    with open(NP_PATH) as f:
        header = next(csv.reader(f))
    return [normalize_neuron_id(n) for n in header[1:]]


def build_standin_pairs_and_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All directed pairs over the real 302-neuron set, labeled 1 if a
    chemical synapse exists in the real White connectome, else 0. This is
    a STAND-IN label for audit purposes only - see module docstring."""
    neurons = load_neuron_set()
    chemical, _gap = load_white_connectome(WHITE_PATH)

    upstream, downstream, label = [], [], []
    for a in neurons:
        for b in neurons:
            if a == b:
                continue
            upstream.append(a)
            downstream.append(b)
            label.append(1 if chemical.get((a, b), 0.0) > 0 else 0)
    return np.array(upstream), np.array(downstream), np.array(label)


def report_fold(name: str, fold_i: int, upstream, downstream, label, train_idx, test_idx, *, both_held_out: bool):
    n_train, n_test = len(train_idx), len(test_idx)
    pos_train = int(label[train_idx].sum())
    pos_test = int(label[test_idx].sum())
    train_up_neurons = set(upstream[train_idx])
    test_up_neurons = set(upstream[test_idx])
    train_down_neurons = set(downstream[train_idx])
    test_down_neurons = set(downstream[test_idx])

    upstream_leak = bool(train_up_neurons & test_up_neurons)
    downstream_leak = bool(train_down_neurons & test_down_neurons)

    print(f"  Fold {fold_i}: n_train={n_train} (pos={pos_train}, neg={n_train - pos_train}), "
          f"n_test={n_test} (pos={pos_test}, neg={n_test - pos_test})")
    print(f"    test-fold unique upstream neurons={len(test_up_neurons)}, "
          f"unique downstream neurons={len(test_down_neurons)}")
    print(f"    upstream neuron overlap train/test: {upstream_leak}  "
          f"downstream neuron overlap train/test: {downstream_leak}")
    if both_held_out:
        assert_no_neuron_leakage(upstream, downstream, train_idx, test_idx)
        print("    [OK] verified: zero neuron leakage (both upstream and downstream unseen in test)")
    else:
        if downstream_leak:
            print("    [expected] downstream neurons DO repeat between train/test - this is the "
                  "known, documented trade-off of leave-stimulated-neuron-out (see splits.py docstring)")


def main():
    upstream, downstream, label = build_standin_pairs_and_labels()
    n = len(upstream)
    print(f"Real pair universe: {n} directed pairs over {len(set(upstream))} real neurons")
    print(f"Stand-in label (chemical synapse present, from real White 1986 data): "
          f"{label.sum()} positive / {n - label.sum()} negative "
          f"({100 * label.mean():.1f}% positive)")
    print("NOTE: this label is a stand-in for audit purposes only. It is NOT the propagation")
    print("target. Real class balance for the actual research question is unknown until")
    print("docs/fetch_propagation_data.md is completed.\n")

    print("=" * 70)
    print("STRICT split: neuron_grouped_folds (both neurons held out)")
    print("=" * 70)
    strict_folds = neuron_grouped_folds(upstream, downstream, n_splits=5, seed=0)
    total_test = 0
    for i, (train_idx, test_idx) in enumerate(strict_folds):
        report_fold("strict", i, upstream, downstream, label, train_idx, test_idx, both_held_out=True)
        total_test += len(test_idx)
    print(f"  Total pairs ever placed in a test fold: {total_test} / {n} "
          f"({100 * total_test / n:.1f}% of all pairs - the rest are dropped every fold "
          f"because they straddle a held-out and a training neuron)\n")

    print("=" * 70)
    print("SECONDARY split: stimulated_neuron_folds (upstream/stimulated neuron held out only)")
    print("=" * 70)
    stim_folds = stimulated_neuron_folds(upstream, n_splits=5, seed=0)
    total_test = 0
    for i, (train_idx, test_idx) in enumerate(stim_folds):
        report_fold("stim-only", i, upstream, downstream, label, train_idx, test_idx, both_held_out=False)
        total_test += len(test_idx)
    print(f"  Total pairs ever placed in a test fold: {total_test} / {n} "
          f"({100 * total_test / n:.1f}% of all pairs - every pair is tested exactly once, "
          f"since only the upstream neuron determines fold membership)\n")

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("The strict both-neurons-held-out split uses roughly 1/n_splits of all pairs as test")
    print("data per fold and drops the rest (straddling pairs) from that fold's training set too -")
    print("this is the correct primary evaluation, but discards real data. The")
    print("stimulated-neuron-only split retains every pair in some fold's test set, at the cost of")
    print("letting downstream neurons repeat between train and test. Both are reported on the real")
    print("propagation target once obtained, with the strict split as the primary, trusted number")
    print("and the stimulated-only split as a secondary, explicitly-weaker comparison - never a")
    print("silent substitute.")


if __name__ == "__main__":
    main()
