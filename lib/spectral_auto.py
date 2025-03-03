import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lobpcg
from sklearn.neighbors import NearestNeighbors


def estimate_sigma(X, knn_k=10):
    """
    Odhadne vhodné `sigma` pro Gaussovskou afinitní matici.
    Používá 90. percentil k-nejbližších vzdáleností a zajistí, že `sigma` není nulové.
    """
    nn = NearestNeighbors(n_neighbors=knn_k).fit(X)
    distances, _ = nn.kneighbors(X)

    sigma = np.percentile(distances[:, -1], 90)  # 90. percentil vzdáleností

    if sigma <= 1e-6:  # Ochrana proti nulové hodnotě
        sigma = np.mean(distances[:, -1])  # Záložní metoda → průměr místo percentilu
    print(f"Calculated sigma: {sigma}")
    return sigma


def compute_knn_affinity(X, k=10, sigma=None):
    """
    Vytvoří řídkou k-NN Gaussovskou afinitní matici.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nn.kneighbors(X)

    # Pokud `sigma` není zadáno, odhadneme ho
    if sigma == 0:
        sigma = estimate_sigma(X, knn_k=k)

    W = sp.lil_matrix((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(1, k + 1):  # Ignorujeme první (je to bod samotný)
            dist_sq = distances[i, j] ** 2
            W[i, indices[i, j]] = np.exp(-dist_sq / (2.0 * sigma ** 2))

    return W.maximum(W.T).tocsr()  # Symetrizace


def compute_full_affinity(X, sigma=None):
    """
    Spočítá plnou Gaussovskou afinitní matici.
    """
    if sigma == 0:
        sigma = estimate_sigma(X, knn_k=10)  # Odhadneme `sigma`, pokud není zadáno

    n = X.shape[0]
    dist_sq = np.sum(X ** 2, axis=1, keepdims=True) - 2 * np.dot(X, X.T) + np.sum(X ** 2, axis=1)
    W = np.exp(-dist_sq / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0)  # Odstraníme vlastní vazby
    return sp.csr_matrix(W)


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


def spectral_clustering(X, k, sigma=None, use_knn=True, knn_k=10, normalized=True):
    """
    Spectral Clustering s automatickým výběrem `sigma`, pokud není zadáno.
    """
    # 1) Vytvoření afinitní matice
    if use_knn:
        W = compute_knn_affinity(X, k=knn_k, sigma=sigma)
    else:
        W = compute_full_affinity(X, sigma=sigma)

    # 2) Normalizovaný nebo nenormalizovaný Laplacian
    L = compute_normalized_laplacian(W) if normalized else compute_unnormalized_laplacian(W)

    # 3) Kontrola Laplacianu
    print(f"Min L: {L.min()}, Max L: {L.max()}")
    print(f"Obsahuje L NaN? {np.isnan(L.data).any()}")
    print(f"Obsahuje L nekonečna? {np.isinf(L.data).any()}")

    # 4) Výpočet vlastních vektorů pomocí `lobpcg`
    X_init = np.random.rand(L.shape[0], k)
    vals, vecs = lobpcg(L, X_init, largest=False)

    # 5) k-means na vlastních vektorech
    labels, _ = my_kmeans(vecs, k)

    return labels
