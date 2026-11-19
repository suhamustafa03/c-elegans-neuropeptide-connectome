import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from nulls import degree_preserving_rewire


def _make_pair_universe(n_neurons=15):
    neurons = [f"N{i:02d}" for i in range(n_neurons)]
    upstream, downstream = [], []
    for a in neurons:
        for b in neurons:
            if a != b:
                upstream.append(a)
                downstream.append(b)
    return np.array(upstream), np.array(downstream)


def test_rewire_preserves_total_edge_count():
    upstream, downstream = _make_pair_universe(12)
    rng = np.random.default_rng(1)
    label = (rng.random(len(upstream)) < 0.15).astype(int)
    rewired = degree_preserving_rewire(upstream, downstream, label, seed=0)
    assert rewired.sum() == label.sum()


def test_rewire_preserves_out_degree_and_in_degree():
    upstream, downstream = _make_pair_universe(12)
    rng = np.random.default_rng(2)
    label = (rng.random(len(upstream)) < 0.2).astype(int)

    def degrees(lab):
        out_deg, in_deg = {}, {}
        for u, d, l in zip(upstream, downstream, lab):
            if l:
                out_deg[u] = out_deg.get(u, 0) + 1
                in_deg[d] = in_deg.get(d, 0) + 1
        return out_deg, in_deg

    orig_out, orig_in = degrees(label)
    rewired = degree_preserving_rewire(upstream, downstream, label, seed=0)
    new_out, new_in = degrees(rewired)
    assert orig_out == new_out
    assert orig_in == new_in


def test_rewire_actually_changes_which_edges_are_positive():
    upstream, downstream = _make_pair_universe(15)
    rng = np.random.default_rng(3)
    label = (rng.random(len(upstream)) < 0.15).astype(int)
    rewired = degree_preserving_rewire(upstream, downstream, label, seed=0)
    # Degree-preserving does not mean edge-preserving: with enough swap
    # attempts on a reasonably sized graph, the specific edge set should differ.
    assert not np.array_equal(label, rewired)


def test_rewire_no_self_pairs_created():
    upstream, downstream = _make_pair_universe(10)
    rng = np.random.default_rng(4)
    label = (rng.random(len(upstream)) < 0.2).astype(int)
    rewired = degree_preserving_rewire(upstream, downstream, label, seed=0)
    for u, d, l in zip(upstream, downstream, rewired):
        if l:
            assert u != d
