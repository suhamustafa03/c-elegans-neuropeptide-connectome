"""
Baseline predictive models: regularized logistic regression and a
gradient-boosted tree model. No GNN here - the research question is about
whether specific, already-defined connectivity LAYERS improve prediction,
which simple models over the 5 layer-weight features test directly; a GNN
would learn its own representation of graph structure, which is a
different (and, given ~23k pairs and 5 known features, not clearly
justified) question. If a GNN is added later, it should be because a
documented check of sample size/structure justifies it, not by default.

`select_model_nested_cv` implements nested cross-validation: an OUTER
split (from splits.py) gives the held-out test fold; within each outer
training fold, an INNER grouped split is used to select hyperparameters,
so test-fold performance is never used, even indirectly, to pick
hyperparameters.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from splits import neuron_grouped_folds


def make_logistic_regression(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)


def make_gradient_boosted_trees(max_depth: int = 3, n_estimators: int = 100) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=0)


LOGREG_GRID = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
GBT_GRID = [
    {"max_depth": d, "n_estimators": n}
    for d in (2, 3, 4)
    for n in (50, 100)
]


def select_model_nested_cv(
    model_factory,
    param_grid: list[dict],
    X: np.ndarray,
    y: np.ndarray,
    upstream: np.ndarray,
    downstream: np.ndarray,
    train_idx: np.ndarray,
    scorer,
    n_inner_splits: int = 3,
    seed: int = 0,
):
    """Select hyperparameters for `model_factory` using an INNER
    neuron-grouped CV restricted to `train_idx` only (never touching
    whatever outer test fold `train_idx` excludes), then refit the
    best-scoring configuration on all of `train_idx`.

    `scorer(y_true, y_score) -> float`, higher is better. Returns
    (fitted_model, best_params, inner_cv_scores_for_best_params).
    """
    inner_upstream = upstream[train_idx]
    inner_downstream = downstream[train_idx]
    inner_folds = neuron_grouped_folds(inner_upstream, inner_downstream, n_splits=n_inner_splits, seed=seed)

    best_params = None
    best_mean_score = -np.inf
    best_scores = None

    for params in param_grid:
        scores = []
        for inner_train_rel, inner_test_rel in inner_folds:
            if len(inner_test_rel) == 0 or len(inner_train_rel) == 0:
                continue
            inner_train_idx = train_idx[inner_train_rel]
            inner_test_idx = train_idx[inner_test_rel]
            if len(np.unique(y[inner_train_idx])) < 2:
                continue  # can't fit a classifier on a single class
            model = model_factory(**params)
            model.fit(X[inner_train_idx], y[inner_train_idx])
            y_score = model.predict_proba(X[inner_test_idx])[:, 1]
            scores.append(scorer(y[inner_test_idx], y_score))
        if not scores:
            continue
        mean_score = float(np.mean(scores))
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_params = params
            best_scores = scores

    if best_params is None:
        raise RuntimeError(
            "Nested CV could not evaluate any hyperparameter configuration "
            "on this training fold (likely too few neurons/pairs for "
            "n_inner_splits). Reduce n_inner_splits or check the input data."
        )

    final_model = model_factory(**best_params)
    final_model.fit(X[train_idx], y[train_idx])
    return final_model, best_params, best_scores
