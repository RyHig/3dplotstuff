import plotly.graph_objects as go
import numpy as np


def bar_2d():
    fig = go.Figure(
        data=[go.Bar(y=[2, 1, 3])],
        layout_title_text="A Figure Displayed with fig.show()"
    )

    fig.show()


def make_values(start, stop, pts):
    t = np.linspace(start, stop, pts)
    v = (np.cos(t), np.sin(t), t)
    return v


def spiral_3d():
    v = []
    for i in range(5):
        v.append(make_values(5*i, 5*i+10, 50))

    fig = go.Figure(
        data=[go.Scatter3d(
            x=v[0][0],
            y=v[0][1],
            z=v[0][2],
            mode='markers',
            marker=dict(
                size=12,
                color=v[0][2],
                colorscale='Viridis',
                opacity=0.8)
            )],
        layout=go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                         method="animate",
                         args=[None])])]
        ),
        frames=[go.Frame(data=[go.Scatter3d(x=v[1][0], y=v[1][1], z=v[1][2])]),
                go.Frame(data=[go.Scatter3d(x=v[2][0], y=v[2][1], z=v[2][2])]),
                go.Frame(data=[go.Scatter3d(x=v[3][0], y=v[3][1], z=v[3][2])]),
                go.Frame(data=[go.Scatter3d(x=v[4][0], y=v[4][1], z=v[4][2])])]
    )

    fig.show()


spiral_3d()
