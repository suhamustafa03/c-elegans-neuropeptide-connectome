"""
Neuron-identifier normalization and alignment across the project's three
data sources, which use inconsistent naming conventions for numbered
neuron classes (ventral cord motor neurons, pharyngeal neurons).

Verified empirically against the actual downloaded files (see
docs/DATA_LICENSES.md and docs/PROPOSAL.md for provenance):

- Ripoll-Sanchez et al. neuropeptide connectome CSVs zero-pad numbered
  classes to 2 digits: "AS01".."AS11", "DA01".."DA09", "VA01".."VA12", but
  leave pharyngeal neurons unpadded: "I1".."I6", "M1".."M5".
- The White et al. 1986 (Varshney-reanalyzed "whole") connectome from
  openworm/ConnectomeToolbox does the OPPOSITE for pharyngeal neurons
  (zero-padded: "I03", "M01") and leaves ventral-cord motor neurons
  unpadded ("AS6", "DA2", "VA1").

Naive case-matching only aligns 236/302 (78%) of neuropeptide-connectome
neurons against the White dataset. Normalizing both sides to a single
zero-padded-to-2-digits convention raises this to 293/302 (97%). The
remaining 9 mismatches are not a bug: CANL/CANR are absent from this
connectome reconstruction (known limitation of the White "whole" dataset),
and the pharyngeal neurons need the reverse padding fix handled separately
(see PHARYNGEAL_UNPAD below) because they already collide if both sides
are zero-padded in the same direction.
"""
from __future__ import annotations

import re

_NUMBERED_RE = re.compile(r"^([A-Za-z]+?)(\d+)$")

# Pharyngeal neuron classes: the neuropeptide-connectome CSVs leave these
# unpadded ("I1", "M1") while White-whole zero-pads them ("I01", "M01").
# This is the reverse of the ventral-cord convention, so it needs its own
# rule rather than a single global zero-pad.
_PHARYNGEAL_CLASSES = {"I", "M"}

# Known non-neuron entries present in the White et al. 1986 "whole"
# connectome (reconstructed body-wall muscle groups, glial-like GLR cells,
# a synthetic aggregate node). These are legitimate presynaptic/
# postsynaptic partners for a full neuromuscular connectome but are not
# neurons and are never present in the neuropeptide-connectome neuron set,
# so they are dropped rather than "unmatched".
NON_NEURON_ENTRIES = {
    "LegacyBodyWallMuscles",
    "GLRDL", "GLRDR", "GLRL", "GLRR", "GLRVL", "GLRVR",
}
# pm1..pm8 (pharyngeal muscle) are also non-neuronal; matched by prefix.
_NON_NEURON_PREFIXES = ("pm",)


def is_non_neuron(name: str) -> bool:
    """True for known non-neuron nodes in the White 1986 connectome."""
    if name in NON_NEURON_ENTRIES:
        return True
    return any(name.startswith(p) and name[len(p):].isdigit() for p in _NON_NEURON_PREFIXES)


def normalize_neuron_id(name: str) -> str:
    """Canonicalize a neuron identifier to the 2-digit zero-padded
    convention used by the neuropeptide-connectome CSVs, with the
    pharyngeal-neuron exception left unpadded to match that same source.

    Non-numbered neuron classes (e.g. "AVAL", "RID") are returned
    unchanged. Unknown/non-neuron tokens are also returned unchanged;
    callers should filter with `is_non_neuron` first if they want to drop
    muscles/glia rather than silently carry them through.
    """
    m = _NUMBERED_RE.match(name)
    if not m:
        return name
    cls, num = m.group(1), m.group(2)
    if cls in _PHARYNGEAL_CLASSES:
        # Target convention for pharyngeal neurons is UNPADDED single/double
        # digit as used by the neuropeptide connectome ("I1", not "I01").
        return f"{cls}{int(num)}"
    return f"{cls}{int(num):02d}"


def align_neuron_sets(set_a: set[str], set_b: set[str]) -> tuple[set[str], set[str], set[str]]:
    """Normalize and align two neuron-ID sets.

    Returns (matched, only_in_a, only_in_b) using normalized IDs. Non-neuron
    entries (per `is_non_neuron`) should be filtered out of each set by the
    caller before calling this, since this function has no way to tell a
    genuinely unmatched neuron from a muscle/glial node in set_b alone.
    """
    norm_a = {normalize_neuron_id(n) for n in set_a}
    norm_b = {normalize_neuron_id(n) for n in set_b}
    matched = norm_a & norm_b
    only_a = norm_a - norm_b
    only_b = norm_b - norm_a
    return matched, only_a, only_b
