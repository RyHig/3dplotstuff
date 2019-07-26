from statistics import mean
import numpy as np
import plotly.graph_objects as go
import random


def create_dataset(n_of_points, variance, step_size=2, correlation=False):
    val = 1
    y_vals = []
    for i in range(n_of_points):
        y = val + random.randrange(-variance, variance)
        y_vals.append(y)
        if correlation and correlation == 'pos':
            val = val + step_size
        elif correlation and correlation == 'neg':
            val = val - step_size
    x_vals = [i for i in range(len(y_vals))]

    return np.array(x_vals, dtype=np.float64), \
        np.array(y_vals, dtype=np.float64)


def find_best_fit_slope_and_intercept(x_vals, y_vals):
    m = (((mean(x_vals) * mean(y_vals)) - mean(x_vals * y_vals)) /
         (mean(x_vals) * mean(x_vals) - mean(x_vals * x_vals)))
    b = mean(y_vals) - (m * mean(x_vals))
    return m, b


def squared_error(y_vals_original, y_vals_line):
    return sum((y_vals_line - y_vals_original) *
               (y_vals_line - y_vals_original))


def coefficient_of_determination(y_vals_original, y_vals_line):
    # creating an list of the mean y_vals_original data we input.
    y_mean_line = [mean(y_vals_original) for y in y_vals_original]
    # The squared error between the original ys and the line we created.
    # y = mx + b is the line we created earlier
    # when we found the line of best fit and the y intercept
    squared_error_regr = squared_error(y_vals_original, y_vals_line)
    # The squared error between the orignal ys and the mean line.
    squared_error_y_mean = squared_error(y_vals_original, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)
if __name__ == "__main__":
    xs, ys = create_dataset(40, 10, 2, correlation="pos")

    m, b = find_best_fit_slope_and_intercept(xs, ys)
    y_line = m*xs + b

    r_squared = coefficient_of_determination(ys, y_line)

    print(r_squared)

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
