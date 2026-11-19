"""
Null models for assessing whether a connectivity layer's apparent
predictive value is more than what its basic network statistics (in
particular, degree distribution) alone would produce.

Implemented: degree-preserving edge rewiring (a double-edge-swap null).

What this null tests: "if we scrambled WHICH specific neurons are
connected to which, but kept each neuron's total number of connections
(in-degree and out-degree) the same, would a model trained on this
rewired network still look predictive of the real propagation data?" If
yes, the original layer's apparent predictive value may be substantially
explained by degree alone (e.g. "highly-connected neurons tend to be
functionally connected to many things, regardless of which specific
neuron they're paired with") rather than by which specific edges the
connectome layer gets right.

Assumption this null requires to be valid, and when NOT to use it: the
double-edge-swap only preserves degree if the graph is treated as a
single, homogeneous edge set. For a WEIGHTED graph (as the neuropeptide
connectome layers are - real-valued connection strengths, not just
presence/absence), a plain double-edge-swap on the binarized
(weight>0) graph preserves degree but discards weight entirely,
which changes what is being nulled. This implementation binarizes first
and documents that choice; it is appropriate for testing the binary
"connected vs not" feature specifically, but a weighted-preserving null
(e.g. weight-shuffle-among-existing-edges) would be needed to properly
null a feature that uses the continuous weight value, and is NOT
implemented here - do not use this null to claim a continuous-weight
feature's predictive value is "real" without that separate check.

The "expression-shuffled peptide-receptor pairing" null mentioned in the
original project brief is NOT implemented, because it requires access to
the underlying CeNGEN expression / ligand-receptor EC50 tables used to
build the neuropeptide connectome in the first place (present in the
Ripoll-Sanchez repo's "Scripts & data" folder, not yet fetched by this
project) to be done validly - shuffling the already-built adjacency
matrix's edges would not test the same thing (expression-based
construction assumptions) as shuffling the ligand-receptor assignments
that built it. Flagging this as not-yet-implemented rather than
implementing an invalid version of it.
"""
from __future__ import annotations

import numpy as np


def degree_preserving_rewire(
    upstream: np.ndarray,
    downstream: np.ndarray,
    label: np.ndarray,
    n_swaps: int | None = None,
    seed: int = 0,
    max_attempts_factor: int = 10,
) -> np.ndarray:
    """Return a rewired copy of `label` (same length, same number of
    positive edges) produced by repeatedly swapping the endpoints of two
    randomly chosen positive edges, preserving each neuron's in-degree and
    out-degree in the resulting binary directed graph.

    Only rewires among pairs present in (upstream, downstream) - i.e.
    within the actual measured/available pair universe, not among all
    theoretically possible pairs - so the null graph lives on the same
    support as the real data. This matters: rewiring onto pairs that were
    never measured would null against a different, larger pair universe
    than the one actually evaluated.

    `n_swaps` defaults to 10x the number of positive edges, a common rule
    of thumb for adequately randomizing a double-edge-swap null; pass an
    explicit value to mix more or less.
    """
    label = label.copy()
    pos_idx = np.where(label == 1)[0]
    n_pos = len(pos_idx)
    if n_pos < 2:
        return label  # nothing to rewire

    if n_swaps is None:
        n_swaps = 10 * n_pos

    rng = np.random.default_rng(seed)
    current_pos = list(pos_idx)

    # Track each positive row's CURRENT (post-swap) pair explicitly, rather
    # than re-reading the original arrays - a row's pair changes identity
    # as soon as it's involved in any swap, and later swaps must operate
    # on that updated pair, not the original one.
    current_u = {idx: str(upstream[idx]) for idx in current_pos}
    current_d = {idx: str(downstream[idx]) for idx in current_pos}
    pair_set = {(current_u[idx], current_d[idx]) for idx in current_pos}

    attempts = 0
    swaps_done = 0
    max_attempts = n_swaps * max_attempts_factor
    while swaps_done < n_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(len(current_pos), size=2, replace=False)
        idx_i, idx_j = current_pos[i], current_pos[j]
        u_i, d_i = current_u[idx_i], current_d[idx_i]
        u_j, d_j = current_u[idx_j], current_d[idx_j]

        # Degree-preserving swap: (u_i -> d_i) and (u_j -> d_j) become
        # (u_i -> d_j) and (u_j -> d_i).
        new_pair_1 = (u_i, d_j)
        new_pair_2 = (u_j, d_i)

        if u_i == d_j or u_j == d_i:
            continue  # would create a self-pair
        if new_pair_1 in pair_set or new_pair_2 in pair_set:
            continue  # would create a duplicate edge

        pair_set.discard((u_i, d_i))
        pair_set.discard((u_j, d_j))
        pair_set.add(new_pair_1)
        pair_set.add(new_pair_2)
        current_d[idx_i], current_d[idx_j] = d_j, d_i
        swaps_done += 1

    # Translate the final pair_set back into a label array: a pair-index
    # row is positive under the null iff its (upstream, downstream) tuple
    # is in the final pair_set. This only works correctly because the
    # rewiring was restricted to pairs that exist in the real
    # (upstream, downstream) universe, so every pair_set member has a
    # matching row to mark.
    pair_to_idx = {
        (str(u), str(d)): idx for idx, (u, d) in enumerate(zip(upstream, downstream))
    }
    new_label = np.zeros_like(label)
    for pair in pair_set:
        idx = pair_to_idx.get(pair)
        if idx is not None:
            new_label[idx] = 1
    return new_label
