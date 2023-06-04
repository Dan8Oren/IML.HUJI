import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_error = np.zeros(n_learners)
    test_error = np.zeros(n_learners)
    iter_lst = list(range(1, n_learners+1))
    for i in iter_lst:
        train_error[i-1] = model.partial_loss(train_X, train_y, i)
        test_error[i-1] = model.partial_loss(test_X, test_y, i)

    fig = go.Figure(
        data=[go.Scatter(x=iter_lst, y=test_error,
                         name="Train Error", mode="lines"),
              go.Scatter(x=iter_lst, y=train_error,
                         name="Test Error", mode="lines")])
    fig.update_layout(title_text=f"Errors as a function of adaboost iterations",
                      xaxis_title="Iteration",
                      yaxis_title="Error",
                      height=500)
    fig.write_image(f"ex4_1-{noise}.png")
    # fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=1, cols=len(T),
                        subplot_titles=[f"Adaboost fit over {t} models" for t in T])
    fig.update_layout(height=400, width=(400*len(T)), showlegend=False)
    for index, iterations in enumerate(T):
        traces = [decision_surface(lambda X: model.partial_predict(X,
                                                                   iterations),
                                   lims[0], lims[1], density=60,
                                   showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=test_y))]
        fig.add_traces(traces, rows=1, cols=index+1)

    fig.write_image(f"ex4_2-{noise}.png")
    # fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_num_of_iterations = np.argmin(test_error) + 1
    fig = go.Figure([
        decision_surface(lambda X: model.partial_predict(X, best_num_of_iterations), lims[0],
                         lims[1], density=60, showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                   showlegend=False,
                   marker=dict(color=test_y))])
    fig.update_layout(width=500, height=500,  showlegend=False,
                         title=f"Lowest test error ensemble size: {best_num_of_iterations},<br>"
                               f"Accuracy: {1 - test_error[best_num_of_iterations - 1]}")
    fig.write_image(f"ex4_3-{noise}.png")

    # Question 4: Decision surface with weighted samples
    size_factor = 10
    if noise == 0:
        size_factor = 50

    fig = go.Figure(
        [decision_surface(model.predict, lims[0], lims[1], density=60,
                          showscale=False, colorscale=[custom[0], custom[-1]]),
    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                showlegend=False,marker=dict(color=(train_y == 1).astype(int),
                                size=size_factor * model.D_ / np.max(model.D_),
                                symbol=class_symbols[train_y.astype(int)],
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))
         ])
    fig.update_layout(dict1=dict(width=500, height=500,
                                 title="Decision Surface of best ensemble"))
    fig.write_image(f"ex4_4-{noise}.png")
    # fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
