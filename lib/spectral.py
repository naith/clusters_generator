import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lobpcg
from sklearn.neighbors import NearestNeighbors


def compute_knn_affinity(X, k=10, sigma=1.0):
    """
    Vytvoří řídkou k-NN Gaussovskou afinitní matici.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    distances, indices = nn.kneighbors(X)

    W = sp.lil_matrix((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(1, k + 1):  # Ignorujeme první (je to bod samotný)
            dist_sq = distances[i, j] ** 2
            W[i, indices[i, j]] = np.exp(-dist_sq / (2.0 * sigma ** 2))

    return W.maximum(W.T).tocsr()  # Symetrizace


def compute_full_affinity(X, sigma=1.0):
    """
    Spočítá plnou Gaussovskou afinitní matici s řídkým uložením.
    """
    n = X.shape[0]
    dist_sq = np.sum(X ** 2, axis=1, keepdims=True) - 2 * np.dot(X, X.T) + np.sum(X ** 2, axis=1)
    W = np.exp(-dist_sq / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0)  # Odstraníme vlastní vazby
    return sp.csr_matrix(W)  # Řídká matice


def compute_normalized_laplacian(W):
    """
    Vypočítá symetrický normalizovaný Laplacian: L_sym = D^{-1/2} (D - W) D^{-1/2}.
    """
    d = np.array(W.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    L = sp.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return L


def compute_unnormalized_laplacian(W):
    """
    Nenormalizovaný Laplacian: L = D - W.
    """
    d = np.array(W.sum(axis=1)).flatten()
    D = sp.diags(d)
    return D - W


def my_kmeans(X, k, max_iters=100, tol=1e-4):
    """
    k-means s inicializací k-means++.
    """
    n, d = X.shape
    centers = X[np.random.choice(n, 1)]  # První centroid
    for _ in range(k - 1):
        dists = np.min(np.linalg.norm(X[:, None] - centers, axis=2), axis=1)
        probs = dists ** 2 / np.sum(dists ** 2)
        new_center = X[np.random.choice(n, p=probs)]
        centers = np.vstack([centers, new_center])

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        dist_sq = np.linalg.norm(X[:, None] - centers, axis=2) ** 2
        labels = np.argmin(dist_sq, axis=1)

        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.sum((centers - new_centers) ** 2) < tol:
            break
        centers = new_centers

    return labels, centers


def spectral_clustering(X, k, sigma=1.0, use_knn=True, knn_k=10, normalized=True):
    """
    Rychlá verze Spectral Clusteringu s řídkými maticemi a optimalizovaným výpočtem vlastních vektorů.
    """
    # 1) k-NN afinitní matice nebo plná afinitní matice
    if use_knn:
        W = compute_knn_affinity(X, k=knn_k, sigma=sigma)
    else:
        W = compute_full_affinity(X, sigma=sigma)

    # 2) Kontrola W
    print(f"Nejmenší hodnota W: {W.min()}, Největší hodnota W: {W.max()}")
    print(f"Počet nenulových hodnot v W: {W.nnz}")

    # 3) Normalizovaný Laplacian
    L = compute_normalized_laplacian(W) if normalized else compute_unnormalized_laplacian(W)

    # 4) Kontrola Laplacianu
    print(f"Minimální hodnota L: {L.min()}, Maximální hodnota L: {L.max()}")
    print(f"Obsahuje L NaN? {np.isnan(L.data).any()}")
    print(f"Obsahuje L nekonečna? {np.isinf(L.data).any()}")


    # 5) Použití stabilnějšího solveru (lobpcg místo eigsh)
    X_init = np.random.rand(L.shape[0], k)
    vals, vecs = lobpcg(L, X_init, largest=False)

    # 6) k-means na vlastních vektorech
    labels, _ = my_kmeans(vecs, k)

    return labels
