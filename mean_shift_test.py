### mean_shift_test.py
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_blobs
from lib.mean_shift import MeanShift


def test_mean_shift():
    # Generování 2D dat
    X_2D, _ = make_moons(n_samples=300, noise=0.05)
    ms = MeanShift(bandwidth=0.5)
    ms.fit(X_2D)

    # Matplotlib vizualizace
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=ms.labels_, cmap='viridis', s=10)
    plt.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1], c='red', marker='x')
    plt.title("Mean Shift - 2D")
    plt.show()

    # Plotly vizualizace
    fig = px.scatter(x=X_2D[:, 0], y=X_2D[:, 1], color=ms.labels_.astype(str))
    fig.add_trace(go.Scatter(x=ms.cluster_centers_[:, 0], y=ms.cluster_centers_[:, 1], mode='markers',
                             marker=dict(color='red', size=10)))
    fig.show()

    # Generování 3D dat
    X_3D, _ = make_blobs(n_samples=300, centers=3, n_features=3, random_state=42)
    ms_3d = MeanShift(bandwidth=2.0)
    ms_3d.fit(X_3D)

    # Matplotlib 3D vizualizace
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2], c=ms_3d.labels_, cmap='viridis', s=10)
    ax.scatter(ms_3d.cluster_centers_[:, 0], ms_3d.cluster_centers_[:, 1], ms_3d.cluster_centers_[:, 2], c='red',
               marker='x')
    plt.title("Mean Shift - 3D")
    plt.show()

    # Plotly 3D vizualizace
    fig = px.scatter_3d(x=X_3D[:, 0], y=X_3D[:, 1], z=X_3D[:, 2], color=ms_3d.labels_.astype(str))
    fig.add_trace(
        go.Scatter3d(x=ms_3d.cluster_centers_[:, 0], y=ms_3d.cluster_centers_[:, 1], z=ms_3d.cluster_centers_[:, 2],
                     mode='markers', marker=dict(color='red', size=5)))
    fig.show()


test_mean_shift()