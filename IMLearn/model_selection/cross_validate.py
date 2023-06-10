from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    permutation = np.arange(X.shape[0])
    train_scores = np.zeros(cv)
    validation_scores = np.zeros(cv)
    folds = np.array_split(permutation, cv)

    # Perform cross-validation
    for i, fold in enumerate(folds):

        # Split the data into train and validation sets
        train_inx = np.setdiff1d(permutation, fold)
        X_train = X[train_inx]
        y_train = y[train_inx]
        X_val = X[fold]
        y_val = y[fold]

        fitted_estimator = deepcopy(estimator).fit(X_train, y_train)
        y_train_pred = fitted_estimator.predict(X_train)
        y_val_pred = fitted_estimator.predict(X_val)

        # Evaluate the performance using the scoring function
        train_scores[i] = scoring(y_train, y_train_pred)
        validation_scores[i] = scoring(y_val, y_val_pred)

    # Calculate the average scores over all folds
    train_score_avg = np.mean(train_scores)
    validation_score_avg = np.mean(validation_scores)

    return train_score_avg, validation_score_avg
