import pandas as pd
import numpy as np
from sklearn import neighbors, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import plotly.io as pio
import plotly.graph_objects as go
import json, codecs
style.use('fivethirtyeight')

df = pd.read_csv('iris.data')
# this is for use with plotly later for some nice graphs.
z = np.array(df['class'])
df.replace('Iris-setosa', 1, inplace=True)
df.replace('Iris-versicolor', 2, inplace=True)
df.replace('Iris-virginica', 3, inplace=True)

# Just taking the features we're going to plot.
x = np.array([df['petal_length'], df['petal_width']]).T
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# clf = neighbors.KNeighborsClassifier()
# clf.fit(x_train, y_train)

# accuracy = clf.score(x_test, y_test)
# print(accuracy)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
# The columns of the Matrix correspond to the X values.
# The rows correspond to the Y values.
x_step = (x_max - x_min)/Z.shape[1]
y_step = (y_max - y_min)/Z.shape[0]

# turns out ploty.graph_objects is pretty new?
# This works, but similar code in the graph_objects format
# just doesn't work, specifically transforms.
data=[dict(
    type='contour',
    x=np.arange(x_min, x_max+1,x_step),
    y=np.arange(y_min, y_max+1,y_step),
    z=Z,
    # contours=dict(coloring='lines'),
    opacity=0.5,
    line_width=2),
    dict(
    type='scatter',
    x=x[:, 0], 
    y=x[:, 1],
    mode='markers',
    transforms=[dict(
        type='groupby',
        groups=z,
        styles=[
            dict(target='Iris-setosa', value=dict(marker=dict(color='blue'))),
            dict(target='Iris-versicolor', value=dict(marker=dict(color='red'))),
            dict(target='Iris-virginica', value=dict(marker=dict(color='black')))
            ]
    )])
]
fig_dict = dict(data=data)
pio.show(fig_dict, validate=False)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, linewidths=1, edgecolors='k')
plt.show()
