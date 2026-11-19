"""
Imbalance-aware evaluation metrics, with a PROVISIONAL, dependence-aware
bootstrap interval that is explicitly not yet a validated inferential
confidence interval. Read "What this interval does NOT establish" below
before using these numbers in any written claim.

## Why not just resample pairs?

Standard accuracy is misleading on a severely imbalanced target like
"functionally connected" (~6% positive in Randi et al.'s own q<0.05
definition), so AUROC and AUPRC (average precision) are used instead.
Getting any kind of resampling-based interval for either also requires
resampling a unit that respects the data's dependence structure, not
individual pairs.

This dataset is **dyadic**: each observation is indexed by an ORDERED
PAIR drawn from a shared set of neurons, and a neuron recurs across many
pairs (as both sender and receiver). A neuron's own properties
(receptor expression, general excitability, imaging quality for that
specific neuron) plausibly affect every pair it participates in, so
pairs sharing a neuron are not independent observations - exactly the
concern this project's own design documents (`splits.py`) already treat
as decisive for cross-validation. A bootstrap that resamples PAIRS i.i.d.
implicitly assumes each pair carries independent information, which
contradicts that same premise and would understate uncertainty relative
to a method that respects the shared-neuron structure.

## The resampling mechanism: vertex (node) resampling

`dyadic_bootstrap_ci` resamples the NODE (neuron) set with replacement
and reconstructs the implied dyad sample from it, rather than resampling
dyads directly:

1. Draws a bootstrap sample of neurons, with replacement, the same size
   as the original neuron set.
2. Reconstructs the resampled pair set from every ordered pair of DISTINCT
   resampled "slots" (not distinct underlying neurons - the same neuron
   drawn into two different slots contributes to pairs with other slots).
3. Looks up each resampled slot-pair's original observation(s) and
   recomputes the metric on that resampled multiset.

**On the citation:** this vertex-resampling mechanism follows Snijders &
Borgatti (1999, "Non-parametric standard errors and tests for network
statistics", Connections 22(2)), which develops it specifically for
standard errors of NETWORK-LEVEL STATISTICS (e.g. density, degree-based
statistics) computed on a single observed network, under within-network
dependence. **That paper does not address, and its results do not
automatically transfer to, the estimand used here** - the sampling
distribution of a held-out predictive-performance metric (AUROC/AUPRC)
from a model fit by nested cross-validation. The citation justifies the
RESAMPLING MECHANISM as a principled way to target shared-neuron
dependence; it does not by itself establish that the resulting interval
has correct (or even approximately correct) coverage for this different,
model-evaluation estimand. No such coverage check has been done here -
see "What this interval does NOT establish" below.

## Exactly what uncertainty this interval does and does not reflect

`dyadic_bootstrap_ci` takes already-computed `(y_true, y_score)` pairs -
the output of ONE already-fitted model evaluated ONCE on ONE realized
test fold - and resamples only at that stage. Precisely:

- **Captures:** variability in the AUROC/AUPRC statistic that would arise
  from re-measuring the SAME fitted model's performance on a different
  resample of the same underlying neuron set, holding the fitted model
  and the realized train/test split fixed. This is a within-fold,
  post-prediction resampling spread - closer in spirit to "how much could
  this one number wobble just from which specific dyads happened to be
  evaluated" than to a full accounting of modeling uncertainty.
- **Does NOT capture:** variability from refitting the model (e.g. if
  nested CV hyperparameter selection or model fitting were rerun on a
  different resample of the TRAINING data, the resulting model, and its
  performance, would likely differ - this interval holds the fitted model
  fixed throughout and never refits it).
- **Does NOT capture:** variability across different outer
  cross-validation fold assignments (which neurons happen to land in
  train vs. test) - only running and reporting across multiple outer
  folds (or repeats with different seeds) addresses that, separately from
  this function.
- **Does NOT capture:** animal/recording-level measurement dependence
  (see below) or any uncertainty in the underlying data-generating
  process (e.g. calcium-imaging measurement noise) beyond what is already
  baked into the realized `y_true`/`y_score` values.

## What this interval does NOT establish

No ground-truth simulation has been run to check whether this method's
nominal interval (e.g. a 95% percentile interval) achieves anywhere near
its nominal coverage for THIS estimand (nested-CV AUROC/AUPRC on dyadic
network data) - that would require either a theoretical argument this
module does not attempt, or an empirical coverage study on data with a
known ground truth, which has not been done. **Treat the reported bounds
as a provisional, exploratory "bootstrap spread" that correctly targets
shared-neuron dependence in its resampling mechanism, not as a
calibrated inferential confidence interval.** `summarize()` labels its
output accordingly (`"ci_status": "provisional_uncalibrated"`) rather
than calling it a validated CI. Revisit this once real propagation labels
exist: at minimum, re-derive these numbers on the real target variable
rather than the structural stand-in currently used for testing, and
ideally run a coverage check (e.g. simulate data with a known effect and
confirm nominal intervals actually cover it close to the nominal rate)
before treating any specific bound as defensible in a written claim.

## Animal/recording-level dependence: not implemented, not estimated

This accounts for dependence created by shared NEURONS across pairs. It
does NOT account for dependence created by shared ANIMALS or RECORDING
SESSIONS - Randi et al.'s raw data has multiple trials per pair across
113 wild-type animals, and trials from the same animal or session could
share additional correlated noise beyond what shared neuron identity
alone captures. Whether this matters depends on whether the derived
propagation table this project eventually builds is trial-level (with an
animal ID) or already aggregated to one row per pair - not yet known,
since the propagation data has not been obtained (see
docs/fetch_propagation_data.md). `dyadic_bootstrap_ci` accepts an
optional `cluster_ids` extension point for a future two-stage
cluster-then-dyad bootstrap; it currently raises `NotImplementedError`
rather than silently ignoring it, so a caller cannot mistakenly believe
animal-level dependence is being handled.

## What happens if the design can't be determined

If there are too few neurons to support a meaningful resample (fewer
than `min_neurons_for_ci`, default 5), `dyadic_bootstrap_ci` returns NaN
bounds - reporting no interval at all - rather than silently falling
back to a pair-level bootstrap that this module does not consider valid
for reporting, or fabricating a number.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _pairwise_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """INVALID for this project's dyadic data - resamples pairs i.i.d.,
    treating each pair as an independent observation despite neurons
    recurring across many pairs. Kept ONLY as a private comparison point
    (see tests/test_evaluate.py) demonstrating that it understates
    uncertainty relative to `dyadic_bootstrap_ci`. Do not call this from
    `summarize()` or anywhere reporting results."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_scores.append(metric_fn(y_true[idx], y_score[idx]))
    if not boot_scores:
        return float("nan"), float("nan"), float("nan")
    point = metric_fn(y_true, y_score)
    lo = float(np.percentile(boot_scores, 100 * alpha / 2))
    hi = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return point, lo, hi


def dyadic_bootstrap_ci(
    upstream: np.ndarray,
    downstream: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
    min_neurons_for_ci: int = 5,
    cluster_ids: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """PROVISIONAL vertex-resampling bootstrap interval - see the module
    docstring (especially "What this interval does NOT establish") before
    treating the returned bounds as a validated confidence interval.
    `upstream`/`downstream`/`y_true`/`y_score` must all be the same length
    (one entry per pair observation).

    This resamples only the already-computed (y_true, y_score) pairs from
    one already-fitted model on one realized test fold - it does not
    refit any model and does not resample across outer CV folds. See
    "Exactly what uncertainty this interval does and does not reflect" in
    the module docstring.

    `cluster_ids`: NOT YET USED. Reserved for a future two-stage
    cluster-then-dyad bootstrap once the real propagation data's
    animal/recording structure is known. Passing it currently raises
    NotImplementedError rather than silently ignoring it, so a caller
    can't mistakenly believe animal-level dependence is being handled.
    """
    if cluster_ids is not None:
        raise NotImplementedError(
            "cluster_ids (animal/recording-level clustering) is not yet "
            "implemented - the real propagation data's trial/animal "
            "structure is not yet known (see docs/fetch_propagation_data.md). "
            "Do not pass cluster_ids until that structure is confirmed and "
            "this function is extended to use it."
        )

    point = metric_fn(y_true, y_score)

    all_neurons = np.unique(np.concatenate([upstream, downstream]))
    if len(all_neurons) < min_neurons_for_ci:
        return point, float("nan"), float("nan")

    pair_index: dict[tuple, list[int]] = {}
    for idx, (u, d) in enumerate(zip(upstream, downstream)):
        pair_index.setdefault((u, d), []).append(idx)

    rng = np.random.default_rng(seed)
    n_neurons = len(all_neurons)
    boot_scores = []

    for _ in range(n_boot):
        resampled = rng.choice(all_neurons, size=n_neurons, replace=True)
        resampled_rows: list[int] = []
        for i in range(n_neurons):
            ni = resampled[i]
            for j in range(n_neurons):
                if i == j:
                    continue
                key = (ni, resampled[j])
                rows = pair_index.get(key)
                if rows:
                    resampled_rows.extend(rows)
        if not resampled_rows:
            continue
        rows_arr = np.array(resampled_rows)
        yt = y_true[rows_arr]
        if len(np.unique(yt)) < 2:
            continue
        ys = y_score[rows_arr]
        boot_scores.append(metric_fn(yt, ys))

    if not boot_scores:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(boot_scores, 100 * alpha / 2))
    hi = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return point, lo, hi


def summarize(
    upstream: np.ndarray,
    downstream: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    """Full summary for one model/fold's predictions: point estimates and
    a PROVISIONAL bootstrap spread for both AUROC and AUPRC, plus the raw
    class counts (never hide the denominator behind a single summary
    number). Requires upstream/downstream neuron IDs (not just
    labels/scores) because the resampling method needs them.

    The returned `auroc_ci`/`auprc_ci` bounds are NOT a validated
    inferential confidence interval - see `ci_status` in the return value
    and the evaluate.py module docstring for exactly why, what they do
    and don't capture, and what would be needed to validate them."""
    auroc_point, auroc_lo, auroc_hi = dyadic_bootstrap_ci(
        upstream, downstream, y_true, y_score, auroc, n_boot=n_boot, seed=seed
    )
    auprc_point, auprc_lo, auprc_hi = dyadic_bootstrap_ci(
        upstream, downstream, y_true, y_score, auprc, n_boot=n_boot, seed=seed
    )
    return {
        "label": label,
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "n_neurons": int(len(np.unique(np.concatenate([upstream, downstream])))),
        "ci_status": "provisional_uncalibrated",
        "ci_method": (
            "Vertex-resampling bootstrap (mechanism per Snijders & Borgatti 1999, "
            "adapted from network-statistics standard errors to this different "
            "nested-CV predictive-performance estimand - NOT validated for this use; "
            "see evaluate.py module docstring). Captures only within-fold, "
            "post-prediction resampling spread of one already-fitted model's test-fold "
            "metric - NOT variability from model refitting, NOT variability across "
            "outer CV folds, NOT animal/recording-level dependence. Coverage has not "
            "been empirically checked. Do not cite auroc_ci/auprc_ci as a calibrated "
            "confidence interval without revisiting these assumptions on real data."
        ),
        "auroc": auroc_point,
        "auroc_ci": [auroc_lo, auroc_hi],
        "auprc": auprc_point,
        "auprc_ci": [auprc_lo, auprc_hi],
    }
