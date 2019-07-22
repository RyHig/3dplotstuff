from statistics import mean
import numpy as np
import plotly.graph_objects as go


def find_best_fit_slope_and_intercept(x_vals, y_vals):
    m = (((mean(x_vals) * mean(y_vals)) - mean(x_vals * y_vals)) /
         (mean(x_vals) * mean(x_vals) - mean(x_vals * x_vals)))
    b = mean(y_vals) - (m * mean(x_vals))
    return m, b


xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

m, b = find_best_fit_slope_and_intercept(xs, ys)
print(m, b)
y_line = m*xs + b

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=xs,
    y=ys,
    mode='markers'
    )
)
fig.add_trace(go.Scatter(
    x=xs,
    y=y_line
    )
)
fig.show()
