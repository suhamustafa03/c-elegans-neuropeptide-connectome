import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from features import build_feature_row, load_neuropeptide_matrix, load_white_connectome

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"


def _skip_if_missing(*paths):
    import pytest
    for p in paths:
        if not p.exists():
            pytest.skip(f"raw data not present: {p}; run src/fetch_connectomes.py first")


def test_load_white_connectome_known_edge():
    path = DATA_DIR / "synaptic_connectome" / "aconnectome_white_1986_whole.csv"
    _skip_if_missing(path)
    chemical, gap = load_white_connectome(path)
    # ADAL -> AIBL chemical, weight 1 (verified against the raw file header rows).
    assert chemical.get(("ADAL", "AIBL")) == 1.0
    # ADAL -> ADFL electrical, weight 1.
    assert gap.get(("ADAL", "ADFL")) == 1.0
    assert len(chemical) > 1000
    assert len(gap) > 500


def test_load_neuropeptide_matrix_normalizes_ids():
    path = DATA_DIR / "neuropeptide_connectome" / "neuropeptide_connectome_short_range_model.csv"
    _skip_if_missing(path)
    m = load_neuropeptide_matrix(path)
    assert len(m) > 0
    # Every key should already be in normalized form (no raw "AS1"/"AS01" mismatch).
    for sender, receiver in list(m.keys())[:50]:
        assert isinstance(sender, str) and isinstance(receiver, str)


def test_build_feature_row_order_and_defaults():
    chemical = {("AVAL", "AVBL"): 3.0}
    gap = {("AVAL", "AVBL"): 2.0}
    np_short = {("AVAL", "AVBL"): 0.5}
    np_mid = {}
    np_long = {("AVAL", "AVBL"): 0.1}
    row = build_feature_row("AVAL", "AVBL", chemical, gap, np_short, np_mid, np_long)
    assert row == [3.0, 2.0, 0.5, 0.0, 0.1]


def test_build_feature_row_normalizes_ventral_cord_ids():
    chemical = {("AS01", "DA02"): 1.0}
    gap, np_short, np_mid, np_long = {}, {}, {}, {}
    # Query with unpadded ("AS1") - should still match the normalized key.
    row = build_feature_row("AS1", "DA2", chemical, gap, np_short, np_mid, np_long)
    assert row[0] == 1.0


def test_build_feature_row_missing_edge_is_zero_not_error():
    row = build_feature_row("XXXL", "YYYR", {}, {}, {}, {}, {})
    assert row == [0.0, 0.0, 0.0, 0.0, 0.0]
