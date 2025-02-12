import numpy as np
from sklearn.datasets import make_circles, make_moons, make_blobs
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Nastavení pro reprodukovatelnost
np.random.seed(42)

# Generování dat
n_samples = 100
X_circle, y_circle = make_circles(n_samples=n_samples, noise=0.1, factor=0.3)
X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.1)
X_blobs, y_blobs = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0)

# Vytvoření subplotů
fig = make_subplots(rows=1, cols=3,
                    subplot_titles=('make_circles()', 'make_moons()', 'make_blobs()'))

# Circles plot
fig.add_trace(
    go.Scatter(
        x=X_circle[:, 0],
        y=X_circle[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=y_circle,
            colorscale='Viridis',
            showscale=True
        ),
        name='Circles'
    ),
    row=1, col=1
)

# Moons plot
fig.add_trace(
    go.Scatter(
        x=X_moons[:, 0],
        y=X_moons[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=y_moons,
            colorscale='Viridis',
            showscale=True
        ),
        name='Moons'
    ),
    row=1, col=2
)

# Blobs plot
fig.add_trace(
    go.Scatter(
        x=X_blobs[:, 0],
        y=X_blobs[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=y_blobs,
            colorscale='Viridis',
            showscale=True
        ),
        name='Blobs'
    ),
    row=1, col=3
)

# Úprava layoutu
fig.update_layout(
    title='Porovnání generovaných datasetů',
    height=600,
    width=1200,
    showlegend=False,
    template='plotly_white'
)

# Nastavení os
for i in range(1, 4):
    fig.update_xaxes(title_text='Feature 1', row=1, col=i)
    fig.update_yaxes(title_text='Feature 2', row=1, col=i)

fig.show()

# Export do HTML pro sdílení
fig.write_html("generated_datasets.html")