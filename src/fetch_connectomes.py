"""
Download the two small, clearly MIT-licensed connectivity datasets this
project uses directly as source data:

  1. The chemical-synapse + gap-junction connectome: White et al. 1986,
     "whole" variant (Varshney et al. 2011 reanalysis, includes pharynx),
     from the OpenWorm ConnectomeToolbox repository (MIT license).
  2. The short/mid/long-range inferred neuropeptide connectivity matrices
     from Ripoll-Sanchez et al. 2023, from the paper authors' own GitHub
     repository (MIT license).

Both are fetched from pinned file paths on each repo's default branch (not
a pinned commit hash, since these are small, stable, already-published
supplementary files - see docs/DATA_LICENSES.md for the exact URLs, what
license covers each, and why). Re-running this script overwrites any
existing copies, so if a source repo is later updated, this script will
fetch the newer version - check docs/DATA_LICENSES.md's "retrieved on"
dates against the source repos if you need exact reproducibility of a
specific prior run.

This does NOT fetch the Randi et al. propagation data - see
fetch_propagation_data.md for why that is a separate, manual,
not-automatically-run step.

Usage:
    python src/fetch_connectomes.py
"""
from __future__ import annotations

import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

WHITE_URL = (
    "https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/"
    "cect/data/aconnectome_white_1986_whole.csv"
)

NEUROPEPTIDE_BASE = (
    "https://raw.githubusercontent.com/LidiaRipollSanchez/Neuropeptide-Connectome/"
    "main/Adjacency%20matrices%20for%20networks/"
)
NEUROPEPTIDE_FILES = {
    "short": "01022024_neuropeptide_connectome_short_range_model.csv",
    "mid": "01022024_neuropeptide_connectome_mid_range_model.csv",
    "long": "01022024_neuropeptide_connectome_long_range_model.csv",
}


def download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest} ({dest.stat().st_size} bytes)")


def main() -> None:
    download(WHITE_URL, RAW_DIR / "synaptic_connectome" / "aconnectome_white_1986_whole.csv")
    for range_name, filename in NEUROPEPTIDE_FILES.items():
        download(
            NEUROPEPTIDE_BASE + filename,
            RAW_DIR / "neuropeptide_connectome" / f"neuropeptide_connectome_{range_name}_range_model.csv",
        )
    print("Done. See docs/DATA_LICENSES.md for license terms of these files.")


if __name__ == "__main__":
    main()
