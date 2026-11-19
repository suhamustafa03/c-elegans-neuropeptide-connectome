# Does the neuropeptide connectome help predict signal propagation in *C. elegans*?

A computational neuroscience project testing whether an inferred
neuropeptide-connectivity network improves out-of-sample prediction of
*measured* neuron-to-neuron signal propagation in *C. elegans*, beyond
chemical-synapse and gap-junction connectivity alone. Built as an
independent, portfolio/application project, not a validated scientific
publication.

**Full feasibility report and staged plan:** [docs/PROPOSAL.md](docs/PROPOSAL.md)
**Data source verification and license status:** [docs/DATA_LICENSES.md](docs/DATA_LICENSES.md)

## Current status (honest, as of this commit)

**Done and tested:**
- Verified all three primary data sources against their actual repositories
  (not abstracts) — see `docs/PROPOSAL.md` for the feasibility audit.
- Fetched the two cleanly MIT-licensed connectivity layers (synaptic/gap-
  junction from White et al. 1986; short/mid/long-range neuropeptide
  connectivity from Ripoll-Sanchez et al. 2023).
- Neuron-ID alignment across the two conflicting naming conventions found
  in these sources (`src/neuron_id.py`), resolving 300/302 neurons with
  the 2 remaining explained, not silently dropped.
- Leakage-aware, neuron-grouped cross-validation splitter
  (`src/splits.py`) — pairs are only tested on when *both* their neurons
  are held out of training.
- Per-pair feature construction across the 5 connectivity layers
  (`src/features.py`).
- 15 unit tests, all passing (`python -m pytest tests/ -q`), including a
  regression test against the real downloaded connectome files.

**Not yet done:**
- The propagation target data itself (Randi et al. 2023) is not yet
  obtained — its derivation depends on a GPL-3.0 third-party tool run
  outside this repo (see `docs/fetch_propagation_data.md`), and its
  license status meant it is intentionally not committed even once
  derived.
- Models, null analyses, the evaluation harness, and the results website
  do not exist yet.

## Why this project

C. elegans is the only animal with a fully reconstructed synaptic
connectome, yet measured neuron-to-neuron signaling (Randi et al. 2023)
disagrees substantially with what that anatomical connectome predicts.
The authors propose extrasynaptic neuropeptide signaling as part of the
explanation but do not quantitatively test the specific inferred
neuropeptide network (Ripoll-Sanchez et al. 2023) against their own
propagation measurements. This project runs that specific, narrow
comparison properly: as an out-of-sample prediction question, with
neuron-grouped cross-validation (not a leaky random pair split), simple
baselines before anything fancier, and an explicit separation between
"this feature predicts the outcome" and "this feature causes the
outcome."

## Repository structure

```
docs/
  PROPOSAL.md                Full feasibility report, staged plan, scope/limits
  DATA_LICENSES.md           Exact license status of every data source
  fetch_propagation_data.md  Manual, documented (not yet run) steps for the target data
data/
  raw/                       Fetched connectome CSVs (synaptic + neuropeptide; MIT licensed)
  derived/                   Propagation target table once generated locally (gitignored - see above)
src/
  neuron_id.py               Neuron-ID normalization across naming conventions
  features.py                Per-pair feature construction from the 5 connectivity layers
  splits.py                  Leakage-aware, neuron-grouped cross-validation
  fetch_connectomes.py       Downloads the two MIT-licensed connectome layers
tests/
  test_neuron_id.py
  test_features.py
  test_splits.py
```

## Setup and reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/fetch_connectomes.py
python -m pytest tests/ -q
```

This reproduces everything currently built. It does **not** reproduce a
full modeling result, because the propagation target data isn't in the
repo (see above) and the modeling code doesn't exist yet.

## License and attribution

Code in this repository: for the author's own educational/application
use. Data: see [docs/DATA_LICENSES.md](docs/DATA_LICENSES.md) for the
exact, source-by-source license status — do not assume any single license
covers everything in this project.
