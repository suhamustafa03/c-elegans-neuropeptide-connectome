# Does whole-brain neural activity help forecast future behavior in *C. elegans*?

An independent computational neuroscience project. **Completed and run on
real data**, not a validated publication. This README is written so a
reader unfamiliar with the project can get the question, data, methods,
findings, and limitations in a few minutes; full detail is linked
throughout.

## In one paragraph

Using a public whole-brain calcium-imaging dataset of freely behaving *C.
elegans* (Atanas, Kim et al. 2023, *Cell*), this project asks whether
current and recent neural population activity improves **leakage-aware,
time-blocked forecasting** of future locomotor velocity and reversal
onset, beyond a behavior-history-only baseline. It is an explicit
**replication and extension of a specific negative result already
reported in that paper** (Fig. 2D: neural activity could decode past but
not future velocity), not a claim of new discovery. The velocity result
replicates that null finding under a stricter protocol; the reversal-onset
result is more interesting — neural activity carries real, above-baseline
signal, but behavior history remains the stronger predictor and combining
the two adds little. All claims are associational, not causal, and no
connectome or neuropeptide-signaling claim is made (that was a different,
earlier, now-paused direction — see below).

## Research question

> Within individual whole-brain calcium-imaging recordings, does current
> and recent whole-brain neural population activity improve out-of-sample
> forecasting of future locomotor velocity (5s ahead) and reversal onset
> (5s ahead), beyond a behavior-history-only baseline?

## Data

- Atanas, Kim et al. (2023), *Cell* 186(19), 4134–4151.e31.
  DOI: [10.1016/j.cell.2023.07.035](https://doi.org/10.1016/j.cell.2023.07.035).
- Zenodo record [8185377, v2](https://doi.org/10.5281/zenodo.8185377),
  `processed_h5.tar.bz2`, **CC BY 3.0 US** (verified against both the
  rendered record page and its machine-readable API — see
  [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) §4 for the full check,
  including a newer revision that was explicitly **not** used because its
  license metadata is missing).
- **80 real recordings** used (91 files in the archive total; 11 are a
  separate "scrambled-control" set, kept unused). Whole-brain GCaMP
  calcium traces (82–163 neurons per recording, anonymous indices — no
  neuron-identity mapping was usable, see Limitations) with time-aligned
  behavior channels (velocity, reversal state, etc.).
- Raw and extracted recordings are **not committed to this repository**
  (kept in an external, isolated location, per this project's own
  data-handling rule) — only small aggregate result tables are.

## Methods (summary)

- **Forecast horizon:** 5s. **History window:** 10s. **Purge gap:** 15s
  (history + horizon) at every train/test boundary, so no window crosses
  it.
- **Unit of analysis:** the recording (80 total; animal identity is not
  established from the source files).
- **Two evaluation schemes, both reported:** blocked out-of-sample
  (training may include later blocks) and forward-chaining (fully
  prospective, training strictly precedes the test block).
- **Four models per task:** persistence, behavior-history-only,
  neural-population-only, combined. Neural features are population-level
  only (summary statistics + per-recording PCA) — never tied to a fixed
  neuron index.
- **43 marker-free recordings as primary**; **37 marker-present
  recordings as a separate sensitivity analysis**, never pooled.
- Full protocol, exact targets, and the reasoning behind every choice:
  [`docs/BEHAVIOR_FORECAST_RESULTS.md`](docs/BEHAVIOR_FORECAST_RESULTS.md)
  and [`site/content/methods.html`](site/content/methods.html).

## Main findings

| Task | Result |
|---|---|
| Velocity forecasting (5s ahead) | **Null result.** No model beats behavior-history-only (median R² ≈ 0.01); neural-only and combined are often worse than even a persistence baseline. This replicates, under a stricter leakage-aware protocol, the same negative finding Atanas & Kim et al. already reported. |
| Reversal-onset forecasting (5s ahead) | **Real but behavior-dominated signal.** Neural-only clears its own recording's prevalence baseline in every recording where this was checked (42/42 marker-free, 37/37 marker-present — a paired, per-recording comparison, not just a group-median comparison). Behavior-only is the stronger predictor; combining the two beats behavior-only in only ~half of recordings (a near coin-flip), so no claim of reliably added value from combining is made. |

Exact numbers, both CV schemes, both recording groups: `results/behavior_forecast_summary.txt` and `results/behavior_forecast_paired_comparisons.txt`.

## Limitations (stated directly, not buried)

- Associational only — no causal or clinical claim anywhere.
- No neuron-identity mapping was usable, so no connectome-linked or
  per-neuron claim is possible; all neural features are population-level.
- 11.2% of recorded frames were discarded by a conservative segmentation
  rule (18 of 80 recordings had one mid-recording gap each).
- Only one fixed horizon/history window was tested; see
  [`docs/BEHAVIOR_FORECAST_RESULTS.md`](docs/BEHAVIOR_FORECAST_RESULTS.md)
  for whether a sensitivity sweep across horizons was attempted and what
  it found, if anything.
- "Marker-free" recordings are not asserted to be "spontaneous" — that
  stronger claim isn't established by anything inspected.
- A single run of the pipeline (not re-run across random seeds) — model
  fitting itself is deterministic (fixed regularization, no stochastic
  solvers requiring a seed), but this means no refitting-variance
  estimate exists.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m pytest tests/ -q
```

This runs the full test suite (90 tests) on synthetic fixtures and real
connectome CSVs — **no download required**. It does not reproduce the
real-data results above, which need the external, separately-licensed
dataset: see
[`site/content/reproducibility.html`](site/content/reproducibility.html)
for the exact steps (download, isolated environment, pipeline invocation)
to reproduce those from scratch.

## Citation

If you build on this project's code or write-up, cite the underlying data
paper directly (required by its CC BY license), not this repository:

> Atanas, A. A., Kim, J., Wang, Z., et al. Brain-wide representations of
> behavior spanning multiple timescales and states in *C. elegans*. *Cell*
> 186(19), 4134–4151.e31 (2023). https://doi.org/10.1016/j.cell.2023.07.035

## Website

A static results site (`site/`) walks through Overview, Data, Methods,
Results, Interpretation, Status, and Reproducibility. Screenshots:
[`docs/screenshots/`](docs/screenshots/) (captured from the actual
rendered site, not mockups).

## Original (paused) direction: neuropeptide connectome & signal propagation

The project originally asked a different question: whether an inferred
neuropeptide connectome (Ripoll-Sánchez et al. 2023) improves prediction
of *measured* signal propagation (Randi et al. 2023 data), beyond the
synaptic/gap-junction connectome alone — see "Why this project" below for
the original motivation. **This direction is paused, not completed or
abandoned.** The propagation data's reuse terms were ambiguous (full
review in [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) §3), and a
permission request to the corresponding author went unanswered; rather
than use the data without a clear license or wait indefinitely, the
project pivoted to the behavior-forecasting study above. What was built
for this paused direction — **and is explicitly not a completed result**
— is detailed below.

<details>
<summary>Full status of the paused connectome project (click to expand)</summary>

**Done and tested, but only on synthetic/stand-in data — never on real
propagation data, because that data was never obtained:**
- Verified all three primary data sources against their actual
  repositories (not abstracts) — see `docs/PROPOSAL.md`.
- Fetched the two cleanly MIT-licensed connectivity layers (synaptic/gap-
  junction from White et al. 1986; short/mid/long-range neuropeptide
  connectivity from Ripoll-Sánchez et al. 2023).
- Neuron-ID alignment across conflicting naming conventions
  (`src/neuron_id.py`): 300/302 neurons resolved, 2 explained (not
  silently dropped).
- Leakage-aware, neuron-grouped cross-validation splitter (`src/splits.py`)
  and a secondary leave-stimulated-neuron-out split, audited on real
  connectome structure as a stand-in label (`src/audit_splits.py`,
  `results/split_audit_dry_run.txt`): the strict split only ever tests
  19.7% of pairs per fold (to guarantee no leakage); the stimulated-only
  split tests every pair once but lets downstream neurons repeat.
- Per-pair feature construction across the 5 connectivity layers
  (`src/features.py`), baseline models with nested-CV hyperparameter
  selection (`src/models.py`), and a degree-preserving rewiring null
  model (`src/nulls.py`) — a real bug in the null model's edge-tracking
  was caught and fixed by its own unit tests.
- Imbalance-aware evaluation (`src/evaluate.py`): AUROC/AUPRC with a
  **provisional, explicitly uncalibrated** dyadic bootstrap (resamples the
  neuron set, not pairs directly, following Snijders & Borgatti 1999's
  vertex-resampling method for network statistics — that paper does not
  address, and its guarantees do not transfer to, this project's actual
  estimand of nested-CV predictive performance). No coverage simulation
  has been run; the code labels this `"ci_status": "provisional_uncalibrated"`.
- `src/build_pair_dataset.py`, the join between propagation data and the
  connectome layers, is written and tested only against synthetic
  fixtures matching the documented schema.

**Not done, and cannot be done without real propagation data:**
- The propagation target data itself (Randi et al. 2023) was never
  obtained — reuse terms are ambiguous (`docs/DATA_LICENSES.md` §3a), and
  a permission request to the corresponding author went unanswered.
- **No model has been trained on real data and no real predictive result
  exists for this direction.** Everything above is verified-correct
  pipeline infrastructure, not a scientific finding.
- The expression-shuffled peptide-receptor null model (requires CeNGEN
  expression/ligand-receptor tables, not fetched).

</details>

### Why this project (original motivation)

*C. elegans* is the only animal with a fully reconstructed synaptic
connectome, yet measured neuron-to-neuron signaling (Randi et al. 2023)
disagrees substantially with what that anatomical connectome predicts.
The authors propose extrasynaptic neuropeptide signaling as part of the
explanation but do not quantitatively test the specific inferred
neuropeptide network (Ripoll-Sánchez et al. 2023) against their own
propagation measurements. This project set out to run that specific,
narrow comparison properly — as an out-of-sample prediction question,
with neuron-grouped cross-validation, simple baselines first, and an
explicit separation between "this feature predicts the outcome" and
"this feature causes the outcome." It never reached real data (see
above), which is why the active study above exists instead.

## Repository structure

```
docs/
  BEHAVIOR_FORECAST_RESULTS.md  Active study: protocol, data provenance, real results
  PROPOSAL.md                   Paused connectome project: feasibility report, staged plan
  DATA_LICENSES.md              License status of every data source, including the active study's
  fetch_propagation_data.md     Paused project's unexecuted data-fetch steps
  screenshots/                  Real screenshots of the rebuilt site
data/
  raw/                          Paused project's fetched connectome CSVs (MIT licensed)
  derived/                      Paused project's propagation table once generated (gitignored)
results/
  behavior_forecast_summary.txt           Active study: full numeric results
  behavior_forecast_paired_comparisons.txt  Active study: per-recording paired comparisons
  split_audit_dry_run.txt                 Paused project: structural stand-in audit
src/
  behavior_forecast/            Active study: segmentation, features, models, metrics, pipeline, paired comparisons
  neuron_id.py  features.py  splits.py  audit_splits.py      (paused project)
  models.py  evaluate.py  nulls.py  fetch_connectomes.py     (paused project)
  build_pair_dataset.py                                       (paused project)
tests/                          90 tests total
site/                           Results website (see "Website" above)
```

## License and attribution

- **Code in this repository:** no license file is included yet. This is
  an open question for the repository owner (you), not something I've
  decided — see the note left for you in the final handoff for the
  options and tradeoffs.
- **Data:** see [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) for the
  exact, source-by-source license status — do not assume any single
  license covers everything in this project. The active study's own data
  source (Atanas, Kim et al. 2023) requires attribution under its CC BY
  license — see Citation above.
