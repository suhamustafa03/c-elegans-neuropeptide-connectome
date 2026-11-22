# Behavior-forecasting pivot: results

**Status: this is the project's current, active study. The original
neuropeptide-connectome / signal-propagation project (see `PROPOSAL.md`,
`DATA_LICENSES.md`) is PAUSED, not completed or abandoned — it stalled on
unresolved data-reuse licensing (see `DATA_LICENSES.md` §3) and is left
documented as-is for a possible future resumption if licensing is
resolved.**

## Why this project exists

Randi et al.'s stimulus-response propagation atlas — the data the original
project needed — has unresolved reuse terms and was never downloaded or
used. A systematic search for a license-clear substitute led to the Atanas,
Kim et al. (2023, *Cell*) whole-brain calcium-imaging dataset (Zenodo
8185377 v2, **CC BY 3.0 US**, independently verified via both the rendered
page and the API). That dataset is *not* a stimulation-response atlas and
cannot support a connectome-linked propagation question (neuron identities
in the raw recordings cannot be decoded without Julia's JLD2 tooling, which
was explicitly ruled out — see the investigation log in this conversation).
It does, however, support a different, honest question about whether
whole-brain neural activity predicts future behavior.

## Research question

> Within individual whole-brain calcium-imaging recordings of freely
> behaving *C. elegans*, does current and recent whole-brain neural
> population activity improve out-of-sample, time-blocked **forecasting**
> of future locomotor velocity and reversal onset, beyond a
> behavior-history-only baseline?

This is explicitly associational, not causal, and makes no connectome or
synaptic claim.

### Relationship to prior work (not a novel discovery claim)

Atanas & Kim et al. (2023, *Cell*, Fig. 2D) already trained linear decoders
to predict velocity at various time offsets Δt from current neural
activity, and reported that they **could predict past velocity (looking
backward up to ~20s) but could not predict future velocity from current
neural activity**. This project is a **leakage-aware replication and
extension** of that specific negative result, not a new discovery:

- it adds an explicit **behavior-history-only** baseline and a **combined**
  model, to isolate neural activity's *marginal* contribution over
  behavior history specifically (not clearly separated in their Fig. 2D
  decoder comparison);
- it adds an explicit **purge-gapped, leakage-safe, blocked time-series CV**
  protocol with a forward-chaining (fully prospective) variant;
- it extends the forecasting question to **reversal onset** (a rare,
  binary event), which Fig. 2D does not address;
- it uses only **population-level** neural features (never a fixed neuron
  index), since no verified identity mapping links ROI indices across
  recordings in this project (unlike the original paper, which has NeuroPAL
  identities for a subset).

## Data used

- Zenodo 8185377 v2, `processed_h5.tar.bz2`, CC BY 3.0 US, MD5-verified.
- **80 real recordings** + 11 scrambled-control duplicates kept fully
  separate (not used as a negative control here, since their precise role
  was not independently verified and the sensitivity-check scope was
  limited to marker-free vs. marker-present).
- Recordings are labeled **marker-free** (43) or **marker-present** (37)
  based on the presence of a `stim_begin_confocal` timestamp — this is a
  file-metadata distinction, not an independently-verified behavioral
  state. The paper describes 30 heat-stimulus datasets and a separate
  "animals without any stimulus" comparison group, which corroborates but
  does not exactly reconcile with this archive's 37/43 split (the counts
  differ and are not explained by any source checked). **Marker-free
  recordings are therefore not asserted to be "spontaneous"** — only that
  they lack this specific stimulus marker.

### Segmentation and data loss

18 of 80 recordings contain exactly one real-time discontinuity (gap > 3×
that recording's own median sampling interval), in every case splitting
the recording roughly in half. Using only the largest contiguous segment
per recording: **112,251 of 126,351 total frames were kept (14,100
discarded, 11.2%)**.

## Locked evaluation protocol (pre-specified before outcome analysis)

- **Sampling interval:** median ≈ 0.6s/frame, computed per recording from
  actual timestamps (not assumed uniform).
- **Forecast horizon:** h = 5s (≈8 frames, computed per recording).
- **History window:** 10s of trailing behavior/neural features.
- **Purge gap:** 15s (history + horizon) inserted at every train/test
  boundary so no input/target window crosses it.
- **Primary split: blocked out-of-sample prediction** — 5 contiguous
  blocks per recording; training may include blocks chronologically after
  a given test block. **Secondary split: forward-chaining** — training
  restricted to data strictly before the test block (fully prospective;
  first block skipped).
- **Unit of analysis: the recording** (80 of them; animal identity is not
  established from the files). Per-recording metrics are first computed
  per fold, then medianed within the recording, then summarized by
  median/IQR across recordings — never pooled across frames as if
  independent.
- **Scaling/PCA/model fitting** on training folds only, applied (never
  refit) to held-out folds.
- **Reversal-onset metrics:** prevalence (base rate) reported alongside
  PR-AUC and ROC-AUC. **Folds with zero positive or zero negative labels
  in the test set are excluded and counted, never coerced to a PR-AUC of
  0.** One fold was excluded this way in each group (persistence/all
  models share the same excluded-fold set, since exclusion depends only on
  the label distribution).
- **Stimulation policy:** marker-free recordings (43) are the primary
  analysis; marker-present recordings (37) run through the identical
  pipeline as a separate sensitivity analysis, never pooled with the
  primary numbers.

## Results

Full numeric output: [`results/behavior_forecast_summary.txt`](../results/behavior_forecast_summary.txt).

### Velocity forecasting (5s ahead) — a null result, consistent with Atanas et al.

Across both the primary (marker-free, blocked CV) and every other
condition tested, **no model clearly and consistently outperforms the
behavior-history-only baseline**, and neural-activity-only is frequently
*worse* than persistence:

| Model | MAE (median) | R² (median) |
|---|---|---|
| persistence | 0.0530 | −0.573 |
| behavior-only | **0.0463** | **0.011** |
| neural-only | 0.0488 | −0.104 |
| combined | 0.0471 | −0.085 |

*(marker-free, blocked CV; forward-chaining and marker-present give the same qualitative pattern — see the full table.)*

Negative median R² for persistence and neural-only means these models
perform worse than simply predicting the per-fold mean velocity. Only
behavior-history-only reaches a (barely) positive median R². Adding neural
features to behavior history (the "combined" model) does not improve on
behavior-only in any condition tested, and sometimes makes it worse. **This
replicates, under a leakage-aware protocol, Atanas et al.'s own reported
finding that current neural activity does not forecast future velocity** —
we treat this null result as a meaningful, reportable finding, not a
failure of the pipeline.

### Reversal-onset forecasting (5s ahead) — a positive, more informative result

| Model | PR-AUC (median) | ROC-AUC (median) |
|---|---|---|
| prevalence/persistence | 0.196 | 0.269 |
| behavior-only | 0.522 | 0.818 |
| neural-only | **0.396** | **0.714** |
| combined | **0.553** | **0.823** |

*(marker-free, blocked CV; median prevalence ≈ 0.196, i.e. reversal onsets are a minority class, consistent with treating this as a rare-event problem.)*

- **Neural activity alone carries real, above-baseline signal** for
  forecasting reversal onset. This is checked two ways: by group medians
  (PR-AUC ~0.40 vs. a ~0.20 prevalence baseline, above) and, more
  rigorously, by a **paired, per-recording comparison**
  (`src/behavior_forecast/paired_comparison.py`,
  `results/behavior_forecast_paired_comparisons.txt`): neural-only's
  PR-AUC exceeded that same recording's own prevalence rate in **every
  recording where the comparison was defined — 42/42 marker-free and
  37/37 marker-present, in both CV schemes**. This paired result, not
  just the group-median comparison, is the basis for calling the signal
  real.
- **Behavior history alone is the stronger predictor, and combining does
  not reliably add value.** The same paired check for combined vs.
  behavior-only shows a win count close to a coin flip across the four
  conditions, with a median per-recording difference of only about
  ±0.01 PR-AUC in either direction. **The exact win count is not stable
  across reruns of the identical pipeline on the identical data** — see
  `results/behavior_forecast_paired_comparisons.txt` for both runs' exact
  counts and why (the true effect size is close enough to zero that
  logistic-regression solver-level floating-point noise flips a few
  recordings' win/loss each run). **No formal significance test was run
  on this difference.** The accurate claim is therefore: "neural activity
  alone clears its own baseline in every recording tested (this count
  *does* reproduce exactly), but adding it to behavior history does not
  reliably improve on behavior history alone (roughly half the
  recordings, not a stable fraction)" — not "neural activity makes an
  independent, statistically confirmed contribution," and not "neural
  activity drives reversal prediction."
- **The persistence baseline's ROC-AUC is *below* 0.5 (~0.27).** This is
  not a bug: the onset label explicitly excludes frames already reversing
  at time t, so being in a reversed state at t is, by construction,
  *anti*-correlated with a *new* onset in the next window. Persistence is
  simply the wrong kind of baseline for an onset-specifically label,
  which is why the prevalence rate (not persistence) is the reference
  point for PR-AUC.
- Pattern holds in both the blocked and forward-chaining split, and in
  both the marker-free and marker-present groups — this is not an
  artifact of one CV scheme or one stimulation condition.

## Horizon/history-window sensitivity: considered, not run

A sweep across multiple forecast horizons and history-window lengths
would be a scientifically sound way to check whether the reported
results are specific to the single 5s/10s choice, and `run_recording`'s
`history_seconds`/`horizon_seconds` parameters (`src/behavior_forecast/pipeline.py`)
were made explicit specifically so this sweep could be run without
editing code. **It was not run** for this update: a single full pass
already takes real wall-clock time against an external, non-cached
747 MB download across all 80 recordings and both CV schemes, and even a
small sweep (e.g. 3 horizons) would multiply that cost directly, which
was judged not worth spending for an explicitly optional check versus
finishing the other, higher-priority fixes in this update. This is a
real, stated gap, not a silently skipped one — if you want it run, it
can be, using exactly the parameters above.

## What this project does and does not claim

- Associational only: "activity (or behavior history) improves forecast
  accuracy," never "activity causes behavior" or vice versa.
- No connectome, synaptic, or neuropeptide claim of any kind.
- No claim about animals — all counts are of recordings (80 total, 43/37
  split), since animal identity is not established in these files.
- The velocity null result and the reversal positive result are both
  reported as produced — this file was written only after the pipeline
  ran and is not edited to match an expected conclusion.
