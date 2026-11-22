"""Model definitions for the behavior-forecasting comparison.

Four conditions, for both the velocity-regression and reversal-onset
classification tasks:
  - persistence: carries forward the current value / current-reversal
    state, with no fitting.
  - behavior-history-only: linear/logistic model on BEHAVIOR_HISTORY features.
  - neural-population-only: linear/logistic model on NEURAL_POPULATION features.
  - combined: both feature blocks concatenated.

All scaling and model fitting happens on the training fold only; the fitted
scaler/model are then applied (never refit) to the held-out fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler


def fit_predict_regression(X_train, y_train, X_test):
    scaler = StandardScaler().fit(X_train)
    model = Ridge(alpha=1.0).fit(scaler.transform(X_train), y_train)
    return model.predict(scaler.transform(X_test))


def fit_predict_classification(X_train, y_train, X_test):
    """Returns predicted probability of the positive class. If the training
    fold has only one class present, falls back to a constant score equal
    to the training-set prevalence (cannot fit a usable classifier)."""
    if len(np.unique(y_train)) < 2:
        p = float(y_train.mean())
        return np.full(len(X_test), p)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
    return model.predict_proba(scaler.transform(X_test))[:, 1]


def persistence_velocity(velocity: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Predict v(t+h) = v(t) for each center t."""
    return velocity[centers]


def persistence_reversal(reversal_vec: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Predict reversal-onset score as the current reversal state (1 if
    already reversing at t, else 0) -- a persistence baseline, not the
    prevalence baseline (which is reported separately in metrics)."""
    return reversal_vec[centers].astype(float)
