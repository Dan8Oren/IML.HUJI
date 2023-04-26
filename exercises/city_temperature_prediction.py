import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=["Date"])
    data = data.drop_duplicates().dropna()
    data['DayOfYear'] = data['Date'].dt.dayofyear
    for column in ["Temp", "Month", "Day", "Year"]:
        data = data[data[column] > 0]
    return data


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = df[df["Country"] == 'Israel']
    israel_data["Year"] = israel_data["Year"].astype(str)
    px.scatter(israel_data, x="DayOfYear", y="Temp",
               color="Year",
               labels={'DayOfYear': 'Day Of Year',
                       'Temp': 'Average Daily Temperature'},
               title="Average Daily Temperature In Israel As A Function Of "
                     "The Day "
                     "Of Year").show()
    px.bar(
        israel_data.groupby('Month', as_index=False).agg(std=("Temp", "std")),
        title="Each Month's Standard Deviation Of The Daily Temperatures",
        y="std", x="Month",
        labels={'std': 'Standard Deviation'}).show()

    # Question 3 - Exploring differences between countries
    px.line(df.groupby(['Country', 'Month'], as_index=False).agg(
        mean=("Temp", "mean"), std=("Temp", "std")),
        x="Month", y="mean", color="Country", error_y="std",
        labels={'std': 'Standard Deviation',
                'mean': "Average Temperature"},
        title="Average Temperature By Month").show()

    # Question 4 - Fitting model for different values of `k`
    train, train_response, test, test_response = \
        split_train_test(israel_data["DayOfYear"], israel_data["Temp"],
                         0.75)
    rounded_loss = []
    for i in range(0, 10):
        rounded_loss.append(round(
            PolynomialFitting(i).fit(train.to_numpy(),
                                     train_response.to_numpy())
                .loss(test.to_numpy(), test_response.to_numpy()), 2))
        print(rounded_loss[i], i)

    # Question 5 - Evaluating fitted model on different countries
    polyReg = PolynomialFitting(5) \
        .fit(israel_data["DayOfYear"], israel_data["Temp"])
    countries = df["Country"]
    countries = countries.drop_duplicates()
    bar_data = []
    for country in countries.values:
        if country == "Israel":
            continue
        temp_df = df[df["Country"] == country]
        bar_data.append({"Country": country,
                         "Loss": round(polyReg.loss(temp_df["DayOfYear"],
                                                    temp_df["Temp"]), 2)})

    px.bar(data_frame=bar_data, x="Country", y="Loss", color="Country",
           text="Loss",
           title="Israel Fitted Model's Loss Over Other Countries",
           ).show()
