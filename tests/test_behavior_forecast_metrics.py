import numpy as np
import pytest

from behavior_forecast.metrics import (
    regression_metrics,
    classification_metrics,
    summarize_across_recordings,
)


def test_regression_metrics_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    out = regression_metrics(y_true, y_true.copy())
    assert out["mae"] == 0.0
    assert out["r2"] == 1.0
    assert out["n"] == 4


def test_regression_metrics_nonzero_error():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    out = regression_metrics(y_true, y_pred)
    assert out["mae"] == pytest.approx(2.0 / 3.0)


def test_classification_metrics_defined_case():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    out = classification_metrics(y_true, y_score)
    assert out["undefined"] is False
    assert out["n_pos"] == 2
    assert out["prevalence"] == 0.5
    assert out["roc_auc"] == 1.0
    assert out["pr_auc"] is not None


def test_classification_metrics_undefined_when_no_positives():
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    out = classification_metrics(y_true, y_score)
    assert out["undefined"] is True
    assert out["pr_auc"] is None
    assert out["roc_auc"] is None
    assert out["n_pos"] == 0


def test_classification_metrics_undefined_when_all_positive():
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.6, 0.7, 0.8])
    out = classification_metrics(y_true, y_score)
    assert out["undefined"] is True


def test_classification_metrics_never_reports_pr_auc_zero_when_undefined():
    """The specific failure mode this function exists to prevent: an
    undefined PR-AUC must never silently come back as 0.0."""
    y_true = np.zeros(10, dtype=int)
    y_score = np.random.default_rng(0).random(10)
    out = classification_metrics(y_true, y_score)
    assert out["pr_auc"] != 0.0
    assert out["pr_auc"] is None


def test_summarize_across_recordings_basic_median_and_iqr():
    out = summarize_across_recordings([1.0, 2.0, 3.0, 4.0, 5.0])
    assert out["n"] == 5
    assert out["n_excluded"] == 0
    assert out["median"] == 3.0


def test_summarize_across_recordings_excludes_none_values():
    out = summarize_across_recordings([1.0, None, 3.0, None])
    assert out["n"] == 2
    assert out["n_excluded"] == 2
    assert out["median"] == 2.0


def test_summarize_across_recordings_empty_list_returns_none_median():
    out = summarize_across_recordings([])
    assert out["n"] == 0
    assert out["median"] is None


def test_summarize_across_recordings_all_none_returns_none_median_with_count():
    out = summarize_across_recordings([None, None, None])
    assert out["n"] == 0
    assert out["n_excluded"] == 3
    assert out["median"] is None
