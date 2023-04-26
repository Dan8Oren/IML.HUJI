from typing import NoReturn, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from IMLearn.learners.regressors import LinearRegression
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
    if y is not None:
        X["price"] = y

    data = X.dropna()
    data = data.drop_duplicates()
    data = data.drop(["id", "date", "lat", "long", "sqft_lot15",
                      "sqft_living15"], axis=1)
    data = pd.get_dummies(data, prefix='zipcode_', columns=['zipcode'])
    renovate_and_built = data[["yr_renovated", "yr_built"]]
    data = data.drop(["yr_renovated", "yr_built"], axis=1)
    data["yr_modified"] = renovate_and_built.max(axis=1)

    return data.drop("price", axis=1), data["price"]


def remove_bad_prices(X: pd.DataFrame):
    X = X[X["price"] > 0]
    return X


def process_train_data(X: pd.DataFrame):
    for column in ["bathrooms", "floors", "sqft_lot", "sqft_basement"]:
        X = X[X[column] >= 0]
    for column in ["price", "sqft_living", "sqft_above", "yr_built"]:
        X = X[X[column] > 0]

    return X


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
    data = X
    for feature in data:
        if "zipcode" in feature:
            continue
        feature = X[feature]
        pearson_corr = np.cov(feature, y) / (np.std(feature) * np.std(y))
        pearson_corr = pearson_corr[0][1]
        fig = go.Figure(go.Scatter(x=feature,
                                   y=y),
                        dict(
                            title=f"Correlation between {feature.name} and "
                                  f"the price,<br>"
                                  f"Pearson Correlation: {pearson_corr}",
                            xaxis_title=f"{feature.name} Sample values",
                            yaxis_title="price"))
        fig.update_traces(mode='markers',
                          marker=dict(line_width=1, symbol='circle',
                                      size=6))
        fig.write_image(output_path + f"/{feature.name}-eval.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    df = remove_bad_prices(df)
    # Question 1 - split data into train and test sets
    price = df["price"]
    train, train_price, test, test_price = \
        split_train_test(df, price, .75)

    # Question 2 - Preprocessing of housing prices dataset
    train = process_train_data(train)
    train, train_price = preprocess_data(train, train_price)
    test, test_price = preprocess_data(test, test_price)

    # Question 3 - Feature evaluation with respect to response
    # feature_evaluation(train, train_price, ".\house_pricing_eval")

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
    lr = LinearRegression(True)
    percentages = list(range(10, 101))
    times = 10
    all_loss = np.ndarray((len(percentages), times))
    for index, percent in enumerate(percentages):
        for time in range(times):
            samples = train.sample(frac=percent / 100)
            sample_responses = train_price.loc[samples.index]
            all_loss[index, time] = lr.fit(samples, sample_responses) \
                .loss(test, test_price)

    all_mean_loss = np.mean(all_loss, axis=1)
    all_std_loss = np.std(all_loss, axis=1)
    upper_bound = all_mean_loss + 2 * all_std_loss
    lower_bound = all_mean_loss - 2 * all_std_loss

    trace = go.Scatter(x=percentages, y=all_mean_loss, mode='markers+lines',
                       name='Function', showlegend=False)

    # Create trace for the confidence interval
    upper_line = go.Scatter(x=percentages,
                            y=upper_bound,
                            fill='tonexty',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            mode="lines",
                            showlegend=False,
                            name='Confidence Interval')
    lower_line = go.Scatter(x=percentages,
                            y=lower_bound,
                            fill='none',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            mode="lines",
                            showlegend=False,
                            name='Confidence Interval')

    fig = go.Figure(data=[lower_line, upper_line, trace], layout=go.Layout(
        xaxis=dict(title="Percentage Of Training Samples"),
        yaxis=dict(title="MSE value"),
        title="MSE As A Function Of The Fraction Over The Training Set",
        height=400))
    fig.write_image("endPlot.png")
