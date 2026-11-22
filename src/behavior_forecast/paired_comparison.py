"""Paired, per-recording comparisons supporting two specific claims made in
docs/BEHAVIOR_FORECAST_RESULTS.md and on the site's Results page:

  - "neural-only's reversal-onset PR-AUC exceeds that same recording's own
    prevalence rate" -- checked per recording, not by comparing group
    medians (which can hide whether an effect holds within a recording).
  - "combined does not consistently beat behavior-only" -- same per-recording
    logic, reported as a win count and a median paired difference, with no
    significance test run (none is claimed).

Operates on the per-recording JSON produced by running the pipeline (see
`run_full_analysis.py`, external to this repo since it needs h5py and the
raw recordings); this module itself has no h5py dependency and is fully
testable on small synthetic fixtures with no data download.
"""
from __future__ import annotations

import statistics
from typing import Optional


def _recording_median(rec: dict, scheme: str, model: str, task: str, field: str) -> Optional[float]:
    """Median of `field` across one recording's own folds, skipping folds
    where the metric is undefined. Returns None if no fold has it defined."""
    fold_metrics = rec[scheme]["results"][model][task]
    vals = [fm[field] for fm in fold_metrics
            if fm.get(field) is not None and not fm.get("undefined", False)]
    if not vals:
        return None
    return float(statistics.median(vals))


def paired_win_count(records: list[dict], scheme: str, task: str, field: str,
                      model_a: str, model_b: str) -> dict:
    """For each recording, compare model_a's vs model_b's per-recording
    median metric. Returns win/tie/loss counts for model_a and the median
    signed difference (a - b), over recordings where both are defined.
    Higher `field` assumed better (true for pr_auc, roc_auc, r2; mae is
    handled by the caller negating or by passing higher_is_better=False
    via `direction`)."""
    n_a_gt_b = 0
    n_tie = 0
    n_b_gt_a = 0
    n_excluded = 0
    diffs = []
    for rec in records:
        a = _recording_median(rec, scheme, model_a, task, field)
        b = _recording_median(rec, scheme, model_b, task, field)
        if a is None or b is None:
            n_excluded += 1
            continue
        diffs.append(a - b)
        if a > b:
            n_a_gt_b += 1
        elif b > a:
            n_b_gt_a += 1
        else:
            n_tie += 1
    n_compared = n_a_gt_b + n_tie + n_b_gt_a
    return {
        "scheme": scheme, "task": task, "field": field,
        "model_a": model_a, "model_b": model_b,
        "n_compared": n_compared, "n_excluded": n_excluded,
        "a_gt_b": n_a_gt_b, "tie": n_tie, "b_gt_a": n_b_gt_a,
        "median_diff_a_minus_b": float(statistics.median(diffs)) if diffs else None,
    }


def neural_beats_prevalence(records: list[dict], scheme: str) -> dict:
    """How many recordings have neural-only's reversal PR-AUC strictly
    above that same recording's own prevalence rate (the correct baseline
    for a rare-event PR-AUC, not the persistence baseline -- see Methods)."""
    n_gt = 0
    n_compared = 0
    n_excluded = 0
    for rec in records:
        neural_pr = _recording_median(rec, scheme, "neural_only", "reversal", "pr_auc")
        prevalence = _recording_median(rec, scheme, "neural_only", "reversal", "prevalence")
        if neural_pr is None or prevalence is None:
            n_excluded += 1
            continue
        n_compared += 1
        if neural_pr > prevalence:
            n_gt += 1
    return {
        "scheme": scheme, "n_compared": n_compared, "n_excluded": n_excluded,
        "neural_gt_prevalence": n_gt,
    }


def combined_vs_behavior_only(records: list[dict], scheme: str) -> dict:
    return paired_win_count(records, scheme, "reversal", "pr_auc", "combined", "behavior_only")


def run_all_comparisons(records: list[dict]) -> dict:
    """Both comparisons, both CV schemes, for one recording group (e.g. all
    marker-free or all marker-present recordings)."""
    out = {}
    for scheme in ["blocked", "forward_chaining"]:
        out[scheme] = {
            "neural_vs_prevalence": neural_beats_prevalence(records, scheme),
            "combined_vs_behavior_only": combined_vs_behavior_only(records, scheme),
        }
    return out


def format_report(marker_free_results: dict, marker_present_results: dict) -> str:
    lines = []
    for label, results in [("marker-free (primary)", marker_free_results),
                            ("marker-present (sensitivity)", marker_present_results)]:
        lines.append(f"=== {label} ===")
        for scheme, r in results.items():
            nvp = r["neural_vs_prevalence"]
            cvb = r["combined_vs_behavior_only"]
            lines.append(f"  [{scheme}]")
            lines.append(
                f"    neural-only PR-AUC > own-recording prevalence: "
                f"{nvp['neural_gt_prevalence']}/{nvp['n_compared']} recordings "
                f"(excluded={nvp['n_excluded']})"
            )
            lines.append(
                f"    combined PR-AUC > behavior-only PR-AUC: "
                f"{cvb['a_gt_b']}/{cvb['n_compared']} recordings "
                f"(tie={cvb['tie']}, excluded={cvb['n_excluded']}, "
                f"median diff={cvb['median_diff_a_minus_b']:.4f})"
            )
        lines.append("")
    return "\n".join(lines)
