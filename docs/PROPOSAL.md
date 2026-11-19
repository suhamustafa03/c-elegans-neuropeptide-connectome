# Project Proposal: Does the neuropeptide connectome help predict signal propagation in *C. elegans*?

> **Status note:** this document distinguishes what is *planned* from what
> is *completed*. As of this commit, data sources are verified and
> fetched, neuron-ID alignment is implemented and tested, and the
> leakage-aware CV splitter and feature-construction code are implemented
> and tested. Models, null analyses, the full evaluation harness, and the
> website are **not yet built** — see "Staged plan" for what's next.

## Research question

Does adding an inferred neuropeptide-connectivity network (short-, mid-,
or long-range models) improve out-of-sample prediction of measured
neuron-pair signal propagation beyond chemical-synapse and gap-junction
connectivity alone, and do the three range-specific models perform
differently from each other?

An observational predictive result of this kind does **not** establish
that any specific neuropeptide causes any specific propagation event —
see "Scope and limits" below.

## Primary sources (feasibility-audited, not taken from abstracts)

| Source | What it provides | Access/license status (see `docs/DATA_LICENSES.md` for full detail) |
|---|---|---|
| Randi et al. 2023, *Nature* (verified via the open arXiv preprint v3, since the Nature page itself redirects to a login wall) | The propagation target: 23,427 neuron pairs, 113 wild-type + 18 `unc-31` mutant animals; per pair, response amplitude, sign, kernel timescale, and a q-value defining a binary "functionally connected" (q<0.05) call (6% of pairs, ~1,314) | Raw/processed data on OSF and in `leiferlab/worm-functional-connectivity`/`pumpprobe` — **no clear license on the data itself**; derivation documented in `docs/fetch_propagation_data.md` but not run, and its output is never committed |
| Ripoll-Sanchez et al. 2023, *Neuron*, via `LidiaRipollSanchez/Neuropeptide-Connectome` (GitHub) | Short/mid/long-range inferred neuropeptide connectivity matrices (302×302), from CeNGEN expression + deorphanized ligand-receptor pairs | **MIT licensed** by the data's own author; fetched and committed |
| White et al. 1986 (Varshney 2011 reanalysis, "whole"), via `openworm/ConnectomeToolbox` | Chemical-synapse + gap-junction connectome (309 nodes incl. muscles/glia) | **MIT licensed**; fetched and committed |

## Closest prior work, and what this project adds

Randi et al. (2023) already show that synaptic+gap-junction anatomy
predicts their measured propagation badly (R² < 0 across pairs), and
already propose extrasynaptic/neuropeptide signaling as part of the
explanation — using `unc-31` mutants (defective in dense-core-vesicle
release) as an indirect test, and citing Ripoll-Sanchez et al. once, for a
single hub-neuron observation (AVD), not a systematic comparison. A
targeted search found no paper that quantitatively fits the actual
short/mid/long-range Ripoll-Sanchez connectivity matrices as predictive
features against this specific propagation dataset in a formal,
leakage-aware, out-of-sample benchmark.

**What this project adds:** that specific, narrow comparison — formalized
as a predictive-modeling question with proper grouped cross-validation,
baselines, and null models — not a claim to have discovered that
neuropeptides matter (the original authors already argued that).

## Target variable

Primary target: the binary "functionally connected" call (q<0.05) as
defined by Randi et al. themselves, once the propagation data is obtained
(see `docs/fetch_propagation_data.md`). This is chosen because it matches
the original paper's own operational definition rather than an arbitrary
threshold invented here, and keeps the primary analysis a standard binary
classification problem compatible with the grouped-CV design already
built. Response amplitude (conditional on connection) is a planned
secondary target; this introduces a selection effect (amplitude is only
defined for already-connected pairs) that will be documented, not
hidden, if pursued.

## Evaluation design (implemented)

- **Leakage control:** `src/splits.py` implements neuron-grouped k-fold
  splitting where a pair is only placed in a fold's test set if *both* its
  upstream and downstream neurons are held out of that fold's training
  set; pairs straddling a held-out and a training neuron are dropped from
  that fold rather than risking leakage. Verified with unit tests
  (`tests/test_splits.py`), including a direct leakage-detection
  assertion and a regression check that the grouped train set is strictly
  smaller than a naive (leaky) split would produce.
- **Feature construction:** `src/features.py` builds a 5-layer feature
  vector per pair (chemical synapse, gap junction, neuropeptide
  short/mid/long-range), with missing edges encoded as 0.0 (absence of a
  predicted edge is itself a prediction, not a missing value).
- **Neuron-ID alignment:** `src/neuron_id.py`, handling the two
  conflicting numbering conventions found empirically between sources
  (ventral-cord motor neurons zero-padded one way, pharyngeal neurons the
  other) — resolves 300/302 neuropeptide-connectome neurons against the
  White connectome, with the 2 unresolved (CANL/CANR) explained as
  genuinely absent from this connectome reconstruction, not a bug.

## Scope and limits (documented up front, not discovered later)

- This is an **observational, predictive** comparison. A feature's
  predictive contribution is evidence of statistical association with
  measured propagation, not proof that any specific peptide-receptor pair
  causally produced any specific response. No claim of mechanistic
  validation will be made.
- Only one synaptic/gap-junction connectome (White 1986 "whole") is used,
  versus the original paper's averaging across White 1986 + four
  Witvliet 2021 reconstructions — a documented simplification (see
  `docs/DATA_LICENSES.md` §4), not an attempt to exactly replicate their
  anatomy baseline.
- The propagation target itself is not yet in hand; the whole downstream
  modeling plan depends on successfully deriving it per
  `docs/fetch_propagation_data.md`, which has not been run.

## Staged plan

1. **Done:** data source verification, fetch scripts for the two clean
   MIT-licensed connectome layers, neuron-ID alignment module + tests,
   feature-construction module + tests, leakage-aware grouped-CV splitter
   + tests.
2. **Next (requires running the documented, external, not-yet-executed
   step):** derive the propagation target table per
   `docs/fetch_propagation_data.md`; verify its neuron-ID overlap against
   the connectome layers the same way this document already verified the
   connectome-to-connectome overlap.
3. Build the full modeling dataset (`src/build_pair_dataset.py`, not yet
   written): join target + all 5 feature layers + combinations.
4. Baselines: regularized logistic regression and a tree-based model
   (scikit-learn), evaluated with the grouped CV splitter, nested CV for
   any hyperparameter selection, metrics suited to class imbalance
   (AUROC, AUPRC, calibration), bootstrap confidence intervals.
5. Null models: degree-preserving rewiring of each connectome layer
   (implemented only if/when its assumptions are checked as valid for
   this data, per your original instruction) to establish what predictive
   gain would look like under a structurally-matched but biologically
   meaningless network.
6. Compare single-layer and combined-layer models; report per-range-model
   (short/mid/long) differences specifically, since that was part of the
   original research question.
7. Write up results (this document + README + a results site), with
   explicit uncertainty intervals and an honest limitations section,
   matching the standard set by the EEG sleep-staging project.

## Reproducing this project's current state

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/fetch_connectomes.py   # the two clean, MIT-licensed layers
python -m pytest tests/ -q        # 15 tests, all pass without the propagation data
```

See `docs/fetch_propagation_data.md` for the (manual, unverified-by-me)
next step to obtain the propagation target data.
