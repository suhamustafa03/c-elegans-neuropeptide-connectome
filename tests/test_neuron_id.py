import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from neuron_id import align_neuron_sets, is_non_neuron, normalize_neuron_id

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"


def test_ventral_cord_padding():
    assert normalize_neuron_id("AS6") == "AS06"
    assert normalize_neuron_id("AS01") == "AS01"
    assert normalize_neuron_id("VA1") == "VA01"
    assert normalize_neuron_id("DA09") == "DA09"


def test_pharyngeal_unpadded():
    assert normalize_neuron_id("I03") == "I3"
    assert normalize_neuron_id("I3") == "I3"
    assert normalize_neuron_id("M01") == "M1"
    assert normalize_neuron_id("M1") == "M1"


def test_non_numbered_unchanged():
    assert normalize_neuron_id("AVAL") == "AVAL"
    assert normalize_neuron_id("RID") == "RID"


def test_non_neuron_detection():
    assert is_non_neuron("LegacyBodyWallMuscles")
    assert is_non_neuron("GLRDL")
    assert is_non_neuron("pm1")
    assert not is_non_neuron("AVAL")
    assert not is_non_neuron("AS01")


def test_align_neuron_sets_basic():
    a = {"AS01", "VA02", "AVAL"}
    b = {"AS1", "VA2", "AVAL"}
    matched, only_a, only_b = align_neuron_sets(a, b)
    assert matched == {"AS01", "VA02", "AVAL"}
    assert only_a == set()
    assert only_b == set()


def test_real_data_alignment():
    """Regression test against the actual downloaded files: confirms the
    normalization resolves to exactly 300/302 matches, with only CANL/CANR
    unmatched (documented as absent from the White 1986 'whole'
    reconstruction) and zero unexplained mismatches on either side."""
    np_path = DATA_DIR / "neuropeptide_connectome" / "neuropeptide_connectome_short_range_model.csv"
    white_path = DATA_DIR / "synaptic_connectome" / "aconnectome_white_1986_whole.csv"
    if not np_path.exists() or not white_path.exists():
        import pytest
        pytest.skip("raw data not present; run src/fetch_connectomes.py first")

    with open(np_path) as f:
        header = next(csv.reader(f))
    np_neurons = set(header[1:])

    white_neurons = set()
    with open(white_path) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) >= 2:
                white_neurons.add(row[0])
                white_neurons.add(row[1])
    white_neurons = {n for n in white_neurons if not is_non_neuron(n)}

    matched, only_np, only_white = align_neuron_sets(np_neurons, white_neurons)
    assert only_np == {"CANL", "CANR"}
    assert only_white == set()
    assert len(matched) == 300
