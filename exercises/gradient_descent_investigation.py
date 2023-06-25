import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import custom


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    all_values = []
    all_weights = []

    def callback(solver, weights, val, grad, t, eta, delta):
        all_values.append(val)
        all_weights.append(np.copy(weights))

    return callback, all_values, all_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    all_l1_converge = make_subplots(rows=1, cols=len(etas))
    all_l2_converge = make_subplots(rows=1, cols=len(etas))
    all_l1_converge.update_xaxes(title_text="Iteration")
    all_l1_converge.update_yaxes(title_text="L1 Norm")
    all_l2_converge.update_xaxes(title_text="Iteration")
    all_l2_converge.update_yaxes(title_text="L2 Norm")

    all_l1_converge.update_layout(showlegend=False,
                                  width=1200,
                                  height=500,
                                  title=dict(
                                      text=f"Convergence Rate Of L1 Norm With Different Fixed Learning Rates",
                                      x=0.5,
                                      xanchor='center',
                                      yanchor='top'
                                  ), margin=dict(t=100))
    all_l2_converge.update_layout(showlegend=False,
                                  width=1200,
                                  height=500,
                                  title=dict(
                                      text=f"Convergence Rate Of L2 Norm With Different Fixed Learning Rates",
                                      x=0.5,
                                      xanchor='center',
                                      yanchor='top'
                                  ), margin=dict(t=100))

    min_loss_l1 = np.inf
    min_eta_l1 = None
    min_loss_l2 = np.inf
    min_eta_l2 = None

    for i, eta in enumerate(etas):
        # Question 1
        l1 = L1(init)
        l2 = L2(init)
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(FixedLR(eta), callback=callback)
        gd.fit(l1, None, None)
        l1_grad = plot_descent_path(L1, descent_path=np.array(weights))

        l1_loss = min(values)
        if l1_loss < min_loss_l1:
            min_loss_l1 = l1_loss
            min_eta_l1 = eta

        l1_conv = go.Figure(
            [go.Scatter(x=np.arange(len(values)), y=values, mode="markers")],
            layout=go.Layout(
                yaxis_title="Norm",
                xaxis_title="Iteration"
            ))

        weights.clear()
        values.clear()
        gd.fit(l2, None, None)
        l2_grad = plot_descent_path(L2, descent_path=np.array(weights))
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(l1_grad.data[0], row=1, col=1)
        fig.add_trace(l1_grad.data[1], row=1, col=1)
        fig.add_trace(l2_grad.data[0], row=1, col=2)
        fig.add_trace(l2_grad.data[1], row=1, col=2)
        fig.update_layout(showlegend=False,
            width=1200,
            height=500,
            title=dict(text=f"Fixed Learning Rate Comparison (eta={eta})",
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            annotations=[
                dict(text="L1 Norm", x=0.18, y=1.05, xref="paper",
                     yref="paper", showarrow=False),
                dict(text="L2 Norm", x=0.82, y=1.05, xref="paper",
                     yref="paper", showarrow=False)
            ]
        )
        fig.write_image(f"./ex5-figures/fixed_lr_eta={eta}.jpg")

        l2_loss = min(values)
        if l2_loss < min_loss_l2:
            min_loss_l2 = l2_loss
            min_eta_l2 = eta

        # Question 3
        l2_conv = go.Figure([go.Scatter(x=np.arange(len(values)), y=values, mode="markers")],
                            layout=go.Layout(
                            yaxis_title="Norm",
                            xaxis_title="Iteration"))
        all_l1_converge.add_trace(l1_conv.data[0], row=1, col=i + 1)
        all_l2_converge.add_trace(l2_conv.data[0], row=1, col=i + 1)
        all_l1_converge.add_annotation(
            text=f"eta={eta}",
            x=((2 * i + 1) / (2 * len(etas))),
            y=1.05,
            xref="paper",
            yref="paper",
            showarrow=False
        )
        all_l2_converge.add_annotation(
            text=f"eta={eta}",
            x=((2 * i + 1) / (2 * len(etas))),
            y=1.05,
            xref="paper",
            yref="paper",
            showarrow=False
        )

    all_l1_converge.write_image("./ex5-figures/l1_convergence.jpg")
    all_l2_converge.write_image("./ex5-figures/l2_convergence.jpg")

    # Question 4
    print(f"\tBest eta for L1 norm: {min_eta_l1}, with loss: {min_loss_l1}")
    print(f"\tBest eta for L2 norm: {min_eta_l2}, with loss: {min_loss_l2}")



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lg = LogisticRegression(include_intercept=True,
                            solver=GradientDescent(
                                max_iter=20000, learning_rate=FixedLR(1e-4)),
                            alpha=0.5)
    lg.fit(X_train, y_train)
    label_prob = lg.predict_proba(X_train)

    fpr, tpr, thresholds = roc_curve(y_train, label_prob)
    c = [custom[0], custom[-1]]
    roc_fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(showlegend=False,
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    roc_fig.write_image("./ex5-figures/roc_curve.jpg")
    opt_alpha = round(thresholds[np.argmax(tpr - fpr)], 2)
    print("\tBest alpha threshold: " + str(opt_alpha))
    lg.alpha_ = opt_alpha
    lg_test_error = lg._loss(X_test, y_test)
    print("\tBest lambda models test loss " + str(lg_test_error))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lamdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    best_lam_l1 = None
    best_lam_l2 = None
    min_v_error_l1 = np.inf
    min_v_error_l2 = np.inf
    for lamda in lamdas:
        logistic_reg_l1 = LogisticRegression(solver=GradientDescent(
            max_iter=20000, learning_rate=FixedLR(1e-4)), alpha=0.5,
            include_intercept=True, penalty="l1", lam=lamda)
        train_err2, v_error1 = cross_validate(logistic_reg_l1, X_train,
                                              y_train, misclassification_error)
        if v_error1 < min_v_error_l1:
            min_v_error_l1 = v_error1
            best_lam_l1 = lamda

        logistic_reg_l2 = LogisticRegression(solver=GradientDescent(
            max_iter=20000, learning_rate=FixedLR(1e-4)), alpha=0.5,
            include_intercept=True, penalty="l2", lam=lamda)
        train_err2, v_error2 = cross_validate(logistic_reg_l2, X_train,
                                              y_train, misclassification_error)
        if v_error2 < min_v_error_l2:
            min_v_error_l2 = v_error2
            best_lam_l2 = lamda

    l1_test_loss = LogisticRegression(
        include_intercept=True, penalty="l1", alpha=0.5,
        solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
        lam=best_lam_l1).fit(X_train, y_train).loss(X_test, y_test)

    print("\n\tBest L1 lamda: " + str(best_lam_l1))
    print("\tL1 model test loss: " + str(l1_test_loss))

    l2_test_loss = LogisticRegression(
        include_intercept=True, penalty="l2", solver=GradientDescent(
            max_iter=20000, learning_rate=FixedLR(1e-4)), alpha=0.5,
        lam=best_lam_l2).fit(X_train, y_train).loss(X_test, y_test)

    print("\n\tBest L2 lamda: " + str(best_lam_l2))
    print("\tL2 model test loss: " + str(l2_test_loss))


if __name__ == '__main__':
    np.random.seed(0)
    print("######## Compare Fixed Learning Rates ########")
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    print("\n\n######## Logistic Regression ########")
    fit_logistic_regression()
