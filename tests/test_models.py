import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from evaluate import auroc
from models import (
    GBT_GRID,
    LOGREG_GRID,
    make_gradient_boosted_trees,
    make_logistic_regression,
    select_model_nested_cv,
)
from splits import neuron_grouped_folds


def _synthetic_pair_dataset(n_neurons=30, n_features=5, seed=0):
    """A fully synthetic dataset for testing the pipeline mechanics only -
    NOT a real result. One feature is made informative so nested CV has
    something real to select on; the rest are noise."""
    rng = np.random.default_rng(seed)
    neurons = [f"N{i:03d}" for i in range(n_neurons)]
    upstream, downstream = [], []
    for a in neurons:
        for b in neurons:
            if a != b:
                upstream.append(a)
                downstream.append(b)
    upstream = np.array(upstream)
    downstream = np.array(downstream)
    n = len(upstream)
    X = rng.random((n, n_features))
    # Make label depend mostly on feature 0 so a fitted model should beat chance.
    y = (X[:, 0] + 0.3 * rng.random(n) > 0.9).astype(int)
    return upstream, downstream, X, y


def test_logistic_regression_beats_chance_on_informative_synthetic_data():
    upstream, downstream, X, y = _synthetic_pair_dataset(seed=1)
    outer_folds = neuron_grouped_folds(upstream, downstream, n_splits=3, seed=0)
    train_idx, test_idx = outer_folds[0]
    assert len(np.unique(y[train_idx])) == 2, "fixture should have both classes in train"

    model, best_params, inner_scores = select_model_nested_cv(
        make_logistic_regression, LOGREG_GRID, X, y, upstream, downstream,
        train_idx, scorer=auroc, n_inner_splits=3, seed=0,
    )
    assert best_params in LOGREG_GRID
    if len(test_idx) > 0 and len(np.unique(y[test_idx])) == 2:
        y_score = model.predict_proba(X[test_idx])[:, 1]
        assert auroc(y[test_idx], y_score) > 0.5


def test_gradient_boosted_trees_nested_cv_runs_and_selects_valid_params():
    upstream, downstream, X, y = _synthetic_pair_dataset(seed=2)
    outer_folds = neuron_grouped_folds(upstream, downstream, n_splits=3, seed=0)
    train_idx, _test_idx = outer_folds[0]

    model, best_params, inner_scores = select_model_nested_cv(
        make_gradient_boosted_trees, GBT_GRID, X, y, upstream, downstream,
        train_idx, scorer=auroc, n_inner_splits=3, seed=0,
    )
    assert best_params in GBT_GRID
    assert len(inner_scores) > 0


def test_model_factories_return_fittable_estimators():
    model = make_logistic_regression(C=0.5)
    assert model.C == 0.5
    model2 = make_gradient_boosted_trees(max_depth=2, n_estimators=10)
    assert model2.max_depth == 2
    assert model2.n_estimators == 10
