import pandas as pd
import quandl
import math
import datetime
import pickle
import os
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Adding a value to allow us to force us to run the classifier.
force_clf = False

# open the apikey.txt file and set quandl api_key to the key within.
with open('./apikey.txt', 'rb') as f:
    quandl.ApiConfig.api_key = f.read().decode("utf-8")


# just plotting the original line, and the "prediction"
def plot_prediction(df: pd.DataFrame):
    # instantiate a figure.
    fig = go.Figure()

    # creating a trace using a Scatter (Scatter defaults to a line graph).
    # this trace is the actual data.
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj. Close']))

    # this trace is the "prediction"
    fig.add_trace(go.Scatter(x=df.index, y=df['Forecast']))

    # updating the figure to have a x-axis, y-axis and title text.
    fig.update_layout(
        title=go.layout.Title(
            text='Google stock prediction',
            xref='paper',
            x=0,
            font=dict(
                family='Courier New, monospace',
                size=24
            )
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='Date',
                font=dict(
                    family='Courier New, monospace',
                    size=18
                )
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='Closing Price',
                font=dict(
                    family='Courier New, monospace',
                    size=18
                )
            )
        )
    )
    fig.show()


# querying quandl to get data for our dataframe
df = quandl.get("WIKI/GOOGL")

# initial trimming of the dataframe for data we want
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# creating two new columns for high to low percent change,
# and percent change from open to close
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df["Adj. Open"]) / df['Adj. Open'] * 100

# trimming the data frame to wanted data
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# instantiating a string to allow for easy changing of the variable we train on
forecast_col = 'Adj. Close'

# for the initial training.
# We don't want to have nan values, but we want to still track the outliers.
df.fillna(-99999, inplace=True)
# forecast_out takes 10% of the dataframe which is 3424 in this case.
# so forecast_out will take the ceiling of 342.4 rounding up to 343.
forecast_out = math.ceil(.1*len(df))

# we're going to shift our forecast_col by the 343.
# This allows us to "predict" the next 343 business days.
df['label'] = df[forecast_col].shift(-forecast_out)

print(df.drop(['label'], 1), df.info())

# x will be a matrix of the first 90% of the data
# minus the forecast column.
# x_lately will be last 10% of the matrix
# y will be our forecast column. without the nans.
x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x = x[:-forecast_out]  # x up to the last 10%
x_lately = x[-forecast_out:]  # x_lately is the last 10%
df.dropna(inplace=True)
y = np.array(df['label'])
# then we split our arrays and matrices into random sets of train and test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# dumps the clf into a pickle if linearregression.pickle doesn't exist
if os.path.exists('./linearregression.pickle') and not force_clf:
    pickle_in = open('linearregression.pickle', 'rb')
    clf = pickle.load(pickle_in)
else:
    clf = LinearRegression(n_jobs=-1)
    clf.fit(x_train, y_train)
    with open('linearregression.pickle', 'wb') as f:
        pickle.dump(clf, f)

# returns the r^2 of the prediction.
# or how well our model fits our data.
accuracy = clf.score(x_test, y_test)
# predict the values from the last 10% of the data from our dataframe.
forecast_set = clf.predict(x_lately)

# set our new dataframe column Forecast, our prediction to nans.
df['Forecast'] = np.nan

# A way to find the next day for our prediction.
last_date = df.iloc[-1].name  # takes the latest date from the dataframe
last_unix = last_date.timestamp()  # convert to timestamp
one_day = 86400  # 60s*60minutes*24hours
next_unix = last_unix + one_day  # the next day is the last day + 86400

# stepping the row by one day.
# setting all the values for that row to nan
# except for the forecast column
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    next_unix += one_day

# call our plotting function to see how our prediction did visually.
plot_prediction(df)
