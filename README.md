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
- A secondary **leave-stimulated-neuron-out** split (`src/splits.py`),
  justified by the real experimental design (only a subset of neurons
  were ever optogenetically stimulated) — audited against the strict
  split on the real 302-neuron/90,902-pair universe, using the real White
  1986 connectome as a stand-in label (`src/audit_splits.py`,
  `results/split_audit_dry_run.txt`). Finding: the strict split only ever
  tests 19.7% of all pairs (drops the rest each fold to guarantee no
  leakage); the stimulated-only split tests every pair exactly once but
  lets downstream neurons repeat between train/test. Both will be
  reported once real labels exist, strict as primary.
- Baseline models (regularized logistic regression, gradient-boosted
  trees), nested cross-validation for hyperparameter selection
  (`src/models.py`), and a degree-preserving rewiring null model
  (`src/nulls.py`) — unit-tested on synthetic fixtures (a real bug in the
  null model's edge-tracking was caught and fixed by these tests before
  being trusted).
- Imbalance-aware evaluation (`src/evaluate.py`): AUROC/AUPRC with a
  **provisional, uncalibrated** bootstrap spread — not a plain pair-level
  bootstrap, but also explicitly not yet claimed as a validated
  confidence interval. Neuron-pair data is dyadic (a neuron recurs across
  many pairs), so `dyadic_bootstrap_ci` resamples the *neuron set* with
  replacement and reconstructs the implied pair sample from it, rather
  than resampling pairs directly. The resampling mechanism follows
  Snijders & Borgatti (1999), who developed vertex resampling for
  standard errors of *network statistics* under within-network
  dependence — **that paper does not address, and its guarantees do not
  automatically transfer to, this project's actual estimand**
  (nested-CV AUROC/AUPRC of a fitted classifier). What this interval
  captures: within-fold resampling spread of one already-fitted model's
  test predictions. What it does NOT capture: variability from
  *refitting* the model, variability across different outer CV fold
  assignments, or animal/recording-level dependence (Randi et al. have
  multiple trials per pair across 113 animals — not accounted for until
  the real derived table's trial/animal structure is known; passing
  `cluster_ids` today raises `NotImplementedError` rather than silently
  ignoring it). No coverage simulation has been run to check whether the
  nominal interval is anywhere close to correct for this estimand — the
  output is explicitly labeled `"ci_status": "provisional_uncalibrated"`,
  and `summarize()`'s `ci_method` field spells out all of this inline. A
  hub-neuron test is a **targeted sanity check** (confirms the mechanism
  responds to designed-in dependence in the expected direction), not
  proof of statistical validity — the test's own docstring says so
  explicitly. If too few neurons are present, the bounds are `NaN`
  rather than a fabricated interval. **This should be revisited once real
  propagation labels exist** — re-derive on the real target, and ideally
  run a coverage check, before citing any specific bound as a defensible
  claim.
- 33 unit tests, all passing (`python -m pytest tests/ -q`), including a
  regression test against the real downloaded connectome files and the
  dyadic-vs-naive bootstrap comparison above.

**Not yet done:**
- The propagation target data itself (Randi et al. 2023) is not yet
  obtained — its derivation depends on a GPL-3.0 third-party tool run
  outside this repo (see `docs/fetch_propagation_data.md`), and its
  license status meant it is intentionally not committed even once
  derived. **No model has been trained on real data and no real result
  exists yet** — everything above is pipeline infrastructure, verified
  correct on synthetic/stand-in data.
- The expression-shuffled peptide-receptor null (mentioned in the
  original brief) is not implemented — it requires the underlying CeNGEN
  expression / ligand-receptor tables, not yet fetched; see `src/nulls.py`
  docstring for why shuffling the already-built adjacency matrix instead
  would not validly test the same thing.
- The evaluation harness is not yet wired into an end-to-end runner
  (`src/run_pipeline.py` doesn't exist yet), and the results website does
  not exist yet.

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
results/
  split_audit_dry_run.txt    Real per-fold audit output (stand-in label, see docs/PROPOSAL.md)
src/
  neuron_id.py               Neuron-ID normalization across naming conventions
  features.py                Per-pair feature construction from the 5 connectivity layers
  splits.py                  Leakage-aware, neuron-grouped CV + leave-stimulated-neuron-out
  audit_splits.py            Per-fold audit of both split strategies on real connectome data
  models.py                  Baseline models + nested CV hyperparameter selection
  evaluate.py                Imbalance-aware metrics (AUROC/AUPRC) with bootstrap CIs
  nulls.py                   Degree-preserving rewiring null model
  fetch_connectomes.py       Downloads the two MIT-licensed connectome layers
tests/
  test_neuron_id.py
  test_features.py
  test_splits.py
  test_models.py
  test_evaluate.py
  test_nulls.py
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
