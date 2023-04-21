from typing import NoReturn, Optional

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px

from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a
    single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X = X.drop(["id", "date", "lat", "long", "sqft_lot15",
                "sqft_living15"], axis=1)
    renovate_and_built = X[["yr_renovated", "yr_built"]]
    X = X.drop(["yr_renovated", "yr_built"], axis=1)
    X["yr_modified"] = renovate_and_built.max(axis=1)
    for column in ["bathrooms", "floors", "sqft_lot", "sqft_basement"]:
        X = X[X[column] >= 0]
    for column in ["price", "sqft_living", "sqft_above", "yr_modified"]:
        X = X[X[column] > 0]

    if y is not None:
        y = y.loc[X.index]

    return X.drop("price", axis=1), X["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    price = df["price"]
    train, train_price, test, test_price = \
        split_train_test(df, price, .75)

    # Question 2 - Preprocessing of housing prices dataset
    train, train_price = preprocess_data(train, train_price)


    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(df.drop("price",axis=1), price)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10
    # times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon
    # of size (mean-2*std, mean+2*std)
