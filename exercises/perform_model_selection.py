from __future__ import annotations

import plotly.graph_objects as go
from sklearn import datasets
from sklearn.linear_model import Lasso

from IMLearn.learners.regressors import LinearRegression, RidgeRegression
from IMLearn.metrics import mean_square_error
from IMLearn.model_selection import cross_validate
from utils import *


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select
    the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2)
    # + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model
    # and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best
    fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of
        the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing
    # portions
    X, y = datasets.load_diabetes(return_X_y=True)
    # random n_samples
    train_indexes = np.random.choice(X.shape[0], size=n_samples, replace=False)
    train_x = X[train_indexes]
    test_indexes = np.setdiff1d(list(range(X.shape[0])), train_indexes)
    test_x = X[test_indexes]
    train_y, test_y = y[train_indexes], y[test_indexes]

    # Question 2 - Perform CV for different values of the regularization
    # parameter for Ridge and Lasso regressions
    lambdas_ridge = np.linspace(0.001, 0.5, n_evaluations)
    lambdas_lasso = np.linspace(0.02, 2.5, n_evaluations)
    train_err_ridge = []
    v_error_ridge = []
    train_err_lasso = []
    v_error_lasso = []
    for l_ridge, l_lasso in zip(lambdas_ridge, lambdas_lasso):
        estimator = RidgeRegression(l_ridge)
        train_err, v_error = cross_validate(estimator, train_x, train_y,
                                            mean_square_error, 5)
        train_err_ridge.append(train_err)
        v_error_ridge.append(v_error)
        estimator = Lasso(l_lasso)
        train_err, v_error = cross_validate(estimator, train_x, train_y,
                                            mean_square_error, 5)
        train_err_lasso.append(train_err)
        v_error_lasso.append(v_error)

    fig1 = go.Figure(
        [go.Scatter(y=train_err_ridge, x=lambdas_ridge, mode='markers+lines',
                    name="train error"),
         go.Scatter(y=v_error_ridge, x=lambdas_ridge, mode='markers+lines',
                    name="validation error")],
        layout=go.Layout(
            title="Ridge Validation & Train Error Over 5-Fold Cross-Validation",
            yaxis_title="MSE",
            xaxis_title="Lamda - Regularization parameter"))

    fig2 = go.Figure(
        [go.Scatter(y=train_err_lasso, x=lambdas_lasso, mode='markers+lines',
                    name="train_error"),
         go.Scatter(y=v_error_lasso, x=lambdas_lasso, mode='markers+lines',
                    name="validation error")],
        layout=go.Layout(
            title="Lasso Validation & Train Error Over 5-Fold Cross-Validation",
            yaxis_title="MSE",
            xaxis_title="Lamda - Regularization parameter"))
    fig1.write_image("ridge-Q2.jpg")
    fig2.write_image("lasso-Q2.jpg")

    # Question 3 - Compare best Ridge model, best Lasso model and Least
    # Squares model
    reg_ridge = lambdas_ridge[np.argmin(v_error_ridge)]
    reg_lasso = lambdas_lasso[np.argmin(v_error_lasso)]
    print(f"Ridge's best regularization parameter: {reg_ridge}")
    print(f"Lasso's best regularization parameter: {reg_lasso}")
    print(
        f"Ridge's test error: "
        f"{RidgeRegression(lam=reg_ridge).fit(train_x, train_y).loss(test_x, test_y)}")
    print(
        f"Lasso's test error: "
        f"{mean_square_error(test_y, Lasso(alpha=reg_lasso).fit(train_x, train_y).predict(test_x))}")
    print(
        f"Least-Square's test error:"
        f"{LinearRegression().fit(train_x, train_y).loss(test_x, test_y)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
