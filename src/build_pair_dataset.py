"""
Join a propagation-pairs table against the connectome feature layers to
produce the final modeling dataset.

This is the script docs/fetch_propagation_data.md calls "not yet written."
It is now written and unit-tested (tests/test_build_pair_dataset.py), but
ONLY against a synthetic fixture file matching the documented expected
schema from fetch_propagation_data.md - it has NEVER been run against the
real propagation data, because that data has not been obtained (see
docs/DATA_LICENSES.md: permission has been requested from the
corresponding author and this project is waiting on a reply before
downloading or processing anything).

Expected input schema (one row per measured neuron pair), matching
fetch_propagation_data.md's documented pumpprobe export sketch:
    from_neuron, to_neuron, n_responses, n_trials, response_rate, mean_amplitude
and OPTIONALLY a `q` column (Randi et al.'s significance q-value), if the
real derivation script is able to pull it from pumpprobe/wormdatamodel
directly rather than re-deriving a cruder threshold.

Binary target definition:
  - If a `q` column is present, "functionally connected" = q < 0.05,
    matching Randi et al.'s own definition exactly.
  - If no `q` column is present, this script REFUSES to silently invent
    one from response_rate alone (e.g. "response_rate > 0") - that would
    not be the paper's actual significance-tested definition, and
    presenting it as such would misrepresent the target variable. It
    instead raises a clear error naming this gap, so a future run against
    real data is forced to either get the real q-values or make an
    explicit, documented decision about a substitute threshold - not have
    one silently assumed by this script.

Usage:
    python src/build_pair_dataset.py --propagation data/derived/propagation_pairs.csv \
        --out data/derived/modeling_dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from features import (
    LAYER_NAMES,
    build_feature_row,
    load_neuropeptide_matrix,
    load_white_connectome,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

REQUIRED_PROPAGATION_COLUMNS = {"from_neuron", "to_neuron", "n_responses", "n_trials", "response_rate"}


class MissingSignificanceColumnError(RuntimeError):
    """Raised when the propagation table has no `q` column. See module
    docstring: this script deliberately does not invent a cruder
    threshold in place of Randi et al.'s actual significance test."""


def load_propagation_table(path: pathlib.Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    missing = REQUIRED_PROPAGATION_COLUMNS - set(reader.fieldnames or [])
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Expected at least: {sorted(REQUIRED_PROPAGATION_COLUMNS)}"
        )
    return rows


def build_dataset(
    propagation_rows: list[dict],
    connectome_dir: pathlib.Path = RAW_DIR,
) -> tuple[list[dict], dict]:
    """Returns (rows, summary). Each output row has from_neuron,
    to_neuron, the 5 connectome-layer features, n_responses, n_trials,
    response_rate, mean_amplitude, and connected (binary target, only if
    a `q` column was present in the input).

    `summary` reports coverage/missingness diagnostics that must be
    checked before trusting the dataset: how many propagation pairs
    matched a known neuron on each side, and the resulting class balance
    once a target is defined.
    """
    has_q = "q" in propagation_rows[0]
    if not has_q:
        raise MissingSignificanceColumnError(
            "Propagation table has no 'q' column (Randi et al.'s significance "
            "q-value). Refusing to derive a substitute binary target from "
            "response_rate alone - that would not be the paper's actual "
            "definition. Either obtain real q-values from the derivation "
            "pipeline, or make an explicit, documented decision about a "
            "substitute threshold before calling this function again."
        )

    chemical, gap = load_white_connectome(connectome_dir / "synaptic_connectome" / "aconnectome_white_1986_whole.csv")
    np_short = load_neuropeptide_matrix(connectome_dir / "neuropeptide_connectome" / "neuropeptide_connectome_short_range_model.csv")
    np_mid = load_neuropeptide_matrix(connectome_dir / "neuropeptide_connectome" / "neuropeptide_connectome_mid_range_model.csv")
    np_long = load_neuropeptide_matrix(connectome_dir / "neuropeptide_connectome" / "neuropeptide_connectome_long_range_model.csv")

    out_rows = []
    n_total = len(propagation_rows)
    n_any_connectome_edge = 0
    n_positive = 0

    for row in propagation_rows:
        from_n, to_n = row["from_neuron"], row["to_neuron"]
        features = build_feature_row(from_n, to_n, chemical, gap, np_short, np_mid, np_long)
        connected = 1 if float(row["q"]) < 0.05 else 0
        n_positive += connected
        if any(f != 0.0 for f in features):
            n_any_connectome_edge += 1

        out_row = {"from_neuron": from_n, "to_neuron": to_n}
        for name, value in zip(LAYER_NAMES, features):
            out_row[name] = value
        out_row["n_responses"] = row["n_responses"]
        out_row["n_trials"] = row["n_trials"]
        out_row["response_rate"] = row["response_rate"]
        out_row["mean_amplitude"] = row.get("mean_amplitude", "")
        out_row["q"] = row["q"]
        out_row["connected"] = connected
        out_rows.append(out_row)

    summary = {
        "n_pairs": n_total,
        "n_pairs_with_any_connectome_edge": n_any_connectome_edge,
        "n_pairs_with_no_connectome_edge_at_all": n_total - n_any_connectome_edge,
        "n_positive": n_positive,
        "n_negative": n_total - n_positive,
        "positive_rate": n_positive / n_total if n_total else float("nan"),
    }
    return out_rows, summary


def write_dataset(rows: list[dict], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["from_neuron", "to_neuron", *LAYER_NAMES, "n_responses", "n_trials", "response_rate", "mean_amplitude", "q", "connected"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--propagation", type=pathlib.Path, default=ROOT / "data" / "derived" / "propagation_pairs.csv")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "derived" / "modeling_dataset.csv")
    args = parser.parse_args()

    if not args.propagation.exists():
        print(
            f"{args.propagation} does not exist. This project does not have the real "
            "propagation data yet - see docs/fetch_propagation_data.md and "
            "docs/DATA_LICENSES.md (currently waiting on explicit permission from "
            "the data's corresponding author before downloading or processing it).",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = load_propagation_table(args.propagation)
    out_rows, summary = build_dataset(rows)
    write_dataset(out_rows, args.out)

    print(f"Built {args.out} ({summary['n_pairs']} pairs)")
    print(f"  Pairs with ANY connectome-layer edge: {summary['n_pairs_with_any_connectome_edge']} / {summary['n_pairs']}")
    print(f"  Pairs with NO connectome-layer edge at all: {summary['n_pairs_with_no_connectome_edge_at_all']}")
    print(f"  Positive (connected, q<0.05): {summary['n_positive']} / {summary['n_pairs']} "
          f"({100 * summary['positive_rate']:.1f}%)")


if __name__ == "__main__":
    main()
