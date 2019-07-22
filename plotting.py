import plotly.graph_objects as go
import numpy as np
N = 100


def bar_2d():
    fig = go.Figure(
        data=[go.Bar(y=[2, 1, 3])],
        layout_title_text="A Figure Displayed with fig.show()"
    )
    fig.show()


def make_spiral_values(start):
    t = np.linspace(start, start + 10, 20)
    v = (np.cos(t), np.sin(t), t)
    return v


def scatter_3d_animate(values):
    fig = go.Figure(
        data=[go.Scatter3d(
            x=values[0],
            y=values[1],
            z=values[2],
            mode='markers',
            marker=dict(size=12,
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
        frames=[go.Frame(
            data=[go.Scatter3d(x=k[0], y=k[1], z=k[2])]) for k in v]
    )

    fig.show()


if __name__ == "__main__":
    v = list(map(make_spiral_values, range(N)))
    scatter_3d_animate(v)
