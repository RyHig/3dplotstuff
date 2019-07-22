import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import datetime
def plot_regression(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj. Close']))
    fig.add_trace(go.Scatter(x=df.index, y=df['Forecast']))

    fig.show()

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100 
df['PCT_change'] = (df['Adj. Close'] - df["Adj. Open"]) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)


x = np.array(df.drop(['label'], 1))

x = preprocessing.scale(x)
x = x[:-forecast_out]
x_lately = x[-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]


plot_regression(df)