import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.spectral import spectral_clustering


def generate_two_moons_data(n_samples=300, noise=0.06):
    """
    Generuje 2D 'two moons' dataset.
    """
    np.random.seed(789)
    n_half = n_samples // 2

    theta1 = np.linspace(0, np.pi, n_half)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)
    data1 = np.vstack((x1, y1)).T

    theta2 = np.linspace(0, np.pi, n_half)
    x2 = 1 - np.cos(theta2)
    y2 = -np.sin(theta2) + 0.5
    data2 = np.vstack((x2, y2)).T

    data1 += noise * np.random.randn(n_half, 2)
    data2 += noise * np.random.randn(n_half, 2)

    X = np.vstack([data1, data2])
    return X


def generate_3d_data():
    """
    Vytvoří jednoduchá 3D data se 3 shluky (Gauss).
    """
    np.random.seed(1234)

    # cluster 1
    X1 = np.random.randn(100, 3) * 0.3 + np.array([0, 0, 0])
    # cluster 2
    X2 = np.random.randn(100, 3) * 0.5 + np.array([3, 3, 1])
    # cluster 3
    X3 = np.random.randn(100, 3) * 0.7 + np.array([-2, 2, 4])

    X = np.vstack([X1, X2, X3])
    return X


def main():
    # ---- 1) Two moons (2D) ----
    X_moons = generate_two_moons_data(n_samples=300, noise=0.06)
    labels_moons = spectral_clustering(
        X_moons,
        k=2,
        sigma=0.2,
        use_knn=True,  # k-NN
        knn_k=10,  # 10 sousedů
        normalized=True
    )

    plt.figure(figsize=(6, 6))
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='viridis')
    plt.title("Spectral Clustering (two moons, 2D)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # ---- 2) Gaussovské 3D shluky ----
    X_3d = generate_3d_data()
    labels_3d = spectral_clustering(
        X_3d,
        k=3,
        sigma=1.0,
        use_knn=True,  # i v 3D klidně vyzkoušíme k-NN
        knn_k=20,
        normalized=True
    )

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels_3d, cmap='viridis')

    ax.set_title("Spectral Clustering (3D data)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    main()
