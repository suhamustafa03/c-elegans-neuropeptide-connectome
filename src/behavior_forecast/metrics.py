"""Metrics for the behavior-forecasting pipeline, with explicit handling of
undefined cases rather than silently coercing them to zero."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "n": int(len(y_true)),
    }


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """PR-AUC, AUROC, and the prevalence (base-rate) baseline, with an
    explicit `undefined` flag when the test set has only one class (no
    positives, or no negatives) -- in which case AUROC/PR-AUC are not
    computed and must not be reported as 0.
    """
    n_pos = int(y_true.sum())
    n = int(len(y_true))
    prevalence = n_pos / n if n else None
    if n_pos == 0 or n_pos == n:
        return {
            "n": n, "n_pos": n_pos, "prevalence": prevalence,
            "pr_auc": None, "roc_auc": None, "undefined": True,
        }
    return {
        "n": n, "n_pos": n_pos, "prevalence": prevalence,
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "undefined": False,
    }


def summarize_across_recordings(values: list[float]) -> dict:
    """Median + IQR summary across recordings (the unit of analysis), with
    the count of recordings actually contributing a defined value."""
    vals = np.array([v for v in values if v is not None], dtype=float)
    n_excluded = len(values) - len(vals)
    if len(vals) == 0:
        return {"n": 0, "n_excluded": n_excluded, "median": None, "q1": None, "q3": None}
    return {
        "n": int(len(vals)),
        "n_excluded": int(n_excluded),
        "median": float(np.median(vals)),
        "q1": float(np.percentile(vals, 25)),
        "q3": float(np.percentile(vals, 75)),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
    }
