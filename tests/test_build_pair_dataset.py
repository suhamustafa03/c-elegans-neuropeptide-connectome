"""
Tests for src/build_pair_dataset.py, the join between a propagation-pairs
table and the connectome feature layers.

Every propagation-pairs row used here is SYNTHETIC - hand-written to match
the documented schema in docs/fetch_propagation_data.md, not derived from
or copied from the real Randi et al. data in any way (which this project
does not have - see docs/DATA_LICENSES.md). This file never reads from or
writes to data/derived/.
"""
import csv
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from build_pair_dataset import (
    MissingSignificanceColumnError,
    build_dataset,
    load_propagation_table,
)

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"


def _write_synthetic_csv(tmp_path, rows, fieldnames):
    path = tmp_path / "synthetic_propagation_pairs.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _require_real_connectome_data():
    white = DATA_DIR / "synaptic_connectome" / "aconnectome_white_1986_whole.csv"
    if not white.exists():
        pytest.skip("real connectome data not present; run src/fetch_connectomes.py first")


def test_load_propagation_table_requires_expected_columns(tmp_path):
    path = _write_synthetic_csv(
        tmp_path,
        [{"from_neuron": "AVAL", "to_neuron": "AVBL"}],
        fieldnames=["from_neuron", "to_neuron"],  # missing n_responses, n_trials, response_rate
    )
    with pytest.raises(ValueError, match="missing required columns"):
        load_propagation_table(path)


def test_load_propagation_table_accepts_valid_schema(tmp_path):
    path = _write_synthetic_csv(
        tmp_path,
        [{
            "from_neuron": "AVAL", "to_neuron": "AVBL",
            "n_responses": "3", "n_trials": "10", "response_rate": "0.3",
            "mean_amplitude": "0.5", "q": "0.01",
        }],
        fieldnames=["from_neuron", "to_neuron", "n_responses", "n_trials", "response_rate", "mean_amplitude", "q"],
    )
    rows = load_propagation_table(path)
    assert len(rows) == 1
    assert rows[0]["from_neuron"] == "AVAL"


def test_build_dataset_refuses_missing_q_column(tmp_path):
    """The script must not silently invent a cruder threshold in place of
    Randi et al.'s actual significance test - this is the behavior that
    protects against misrepresenting the target variable."""
    path = _write_synthetic_csv(
        tmp_path,
        [{
            "from_neuron": "AVAL", "to_neuron": "AVBL",
            "n_responses": "3", "n_trials": "10", "response_rate": "0.3",
        }],
        fieldnames=["from_neuron", "to_neuron", "n_responses", "n_trials", "response_rate"],
    )
    rows = load_propagation_table(path)
    with pytest.raises(MissingSignificanceColumnError):
        build_dataset(rows)


def test_build_dataset_joins_real_connectome_layers_with_synthetic_pairs():
    """Uses the real, already-cleared connectome data (MIT-licensed,
    committed to this repo) joined against entirely synthetic
    propagation-pairs rows - never real propagation measurements."""
    _require_real_connectome_data()
    synthetic_rows = [
        # ADAL -> AIBL has a real chemical-synapse edge (weight 1) in the
        # actual White 1986 data - verified separately in test_features.py.
        {"from_neuron": "ADAL", "to_neuron": "AIBL", "n_responses": "5", "n_trials": "10",
         "response_rate": "0.5", "mean_amplitude": "0.2", "q": "0.001"},
        # A pair with no known connectome edge in any layer (synthetic,
        # deliberately implausible neuron pair for this check).
        {"from_neuron": "XXXL", "to_neuron": "YYYR", "n_responses": "0", "n_trials": "8",
         "response_rate": "0.0", "mean_amplitude": "", "q": "0.9"},
    ]
    out_rows, summary = build_dataset(synthetic_rows)

    assert summary["n_pairs"] == 2
    assert summary["n_positive"] == 1  # only the q=0.001 row is < 0.05
    assert summary["n_negative"] == 1
    assert summary["n_pairs_with_any_connectome_edge"] == 1  # only ADAL->AIBL

    adal_row = next(r for r in out_rows if r["from_neuron"] == "ADAL")
    assert adal_row["chemical_synapse"] == 1.0
    assert adal_row["connected"] == 1

    unknown_row = next(r for r in out_rows if r["from_neuron"] == "XXXL")
    assert unknown_row["chemical_synapse"] == 0.0
    assert unknown_row["connected"] == 0


def test_build_dataset_summary_reports_coverage_not_just_counts():
    _require_real_connectome_data()
    synthetic_rows = [
        {"from_neuron": "ADAL", "to_neuron": "AIBL", "n_responses": "1", "n_trials": "1",
         "response_rate": "1.0", "mean_amplitude": "", "q": "0.001"},
    ]
    _, summary = build_dataset(synthetic_rows)
    assert set(summary.keys()) == {
        "n_pairs", "n_pairs_with_any_connectome_edge",
        "n_pairs_with_no_connectome_edge_at_all", "n_positive", "n_negative", "positive_rate",
    }
