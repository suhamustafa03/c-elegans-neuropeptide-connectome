"""
Build per-neuron-pair feature vectors from the connectivity layers:
chemical synapse weight, gap-junction weight, and short/mid/long-range
inferred neuropeptide-connectivity weights. Each layer is looked up for a
given (upstream, downstream) neuron pair using the normalized neuron IDs
from `neuron_id.py`; a pair with no edge in a given layer gets 0.0 for
that layer's feature, not a missing value - absence of a predicted edge
is itself the layer's prediction for that pair.
"""
from __future__ import annotations

import csv
import pathlib

from neuron_id import is_non_neuron, normalize_neuron_id

LAYER_NAMES = [
    "chemical_synapse",
    "gap_junction",
    "neuropeptide_short_range",
    "neuropeptide_mid_range",
    "neuropeptide_long_range",
]


def load_white_connectome(path: pathlib.Path) -> tuple[dict, dict]:
    """Load the White 1986 'whole' edge list into two dicts keyed by
    (upstream, downstream) normalized neuron-ID tuples: one for chemical
    synapse weights, one for gap-junction weights. Gap junctions are
    symmetric in the source file but are stored here exactly as given
    (both directions appear as separate rows in the source already)."""
    chemical: dict[tuple[str, str], float] = {}
    gap: dict[tuple[str, str], float] = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: pre post type synapses
        for row in reader:
            if len(row) < 4:
                continue
            pre, post, conn_type, n_synapses = row[0], row[1], row[2], row[3]
            if is_non_neuron(pre) or is_non_neuron(post):
                continue
            key = (normalize_neuron_id(pre), normalize_neuron_id(post))
            weight = float(n_synapses)
            if conn_type == "chemical":
                chemical[key] = chemical.get(key, 0.0) + weight
            elif conn_type == "electrical":
                gap[key] = gap.get(key, 0.0) + weight
    return chemical, gap


def load_neuropeptide_matrix(path: pathlib.Path) -> dict[tuple[str, str], float]:
    """Load one of the Ripoll-Sanchez short/mid/long-range square adjacency
    CSVs (row neuron, columns = all neurons) into a dict keyed by
    normalized (sender, receiver) tuples. Convention: row = sending
    neuron, column = receiving neuron, matching the CSV's own 'Row' header
    label and the paper's description of a directed peptidergic network."""
    out: dict[tuple[str, str], float] = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        col_neurons = header[1:]
        for row in reader:
            sender = row[0]
            for col_neuron, value in zip(col_neurons, row[1:]):
                try:
                    w = float(value)
                except ValueError:
                    continue
                if w == 0.0:
                    continue
                key = (normalize_neuron_id(sender), normalize_neuron_id(col_neuron))
                out[key] = w
    return out


def build_feature_row(
    upstream: str,
    downstream: str,
    chemical: dict,
    gap: dict,
    np_short: dict,
    np_mid: dict,
    np_long: dict,
) -> list[float]:
    """Return the 5-element feature vector [chemical, gap, np_short,
    np_mid, np_long] for one (upstream, downstream) pair, in the fixed
    order given by LAYER_NAMES. Neuron IDs are normalized internally, so
    callers can pass raw IDs from any of the three source conventions."""
    key = (normalize_neuron_id(upstream), normalize_neuron_id(downstream))
    return [
        chemical.get(key, 0.0),
        gap.get(key, 0.0),
        np_short.get(key, 0.0),
        np_mid.get(key, 0.0),
        np_long.get(key, 0.0),
    ]
