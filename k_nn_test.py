### knn_test.py
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_blobs, make_circles
from lib.k_nn import KNN


def test_knn():
    # Generování 2D dat (two moons)
    X_2D, y_2D = make_moons(n_samples=300, noise=0.1)
    knn = KNN(k=5)
    y_pred = knn.predict(X_2D, y_2D, X_2D)

    # Matplotlib vizualizace
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y_pred, cmap='viridis', s=10)
    plt.title("KNN - 2D")
    plt.show()

    # Plotly vizualizace
    fig = px.scatter(x=X_2D[:, 0], y=X_2D[:, 1], color=y_pred.astype(str))
    fig.show()

    # Generování 2D dat (dvě soustředné kružnice)
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5)
    y_pred_circles = knn.predict(X_circles, y_circles, X_circles)

    plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_pred_circles, cmap='viridis', s=10)
    plt.title("KNN - 2 soustředné kružnice")
    plt.show()

    fig = px.scatter(x=X_circles[:, 0], y=X_circles[:, 1], color=y_pred_circles.astype(str))
    fig.show()

    # Generování 3D dat
    X_3D, y_3D = make_blobs(n_samples=300, centers=3, n_features=3, random_state=42)
    y_pred_3d = knn.predict(X_3D, y_3D, X_3D)

    # Matplotlib 3D vizualizace
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2], c=y_pred_3d, cmap='viridis', s=10)
    plt.title("KNN - 3D")
    plt.show()

    # Plotly 3D vizualizace
    fig = px.scatter_3d(x=X_3D[:, 0], y=X_3D[:, 1], z=X_3D[:, 2], color=y_pred_3d.astype(str))
    fig.show()


test_knn()