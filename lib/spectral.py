import numpy as np


def compute_full_affinity(X, sigma=1.0):
    """
    Spočítá plnou Gaussovskou afinitní matici (každý bod s každým).
    """
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = X[i] - X[j]
            dist_sq = np.dot(diff, diff)
            W[i, j] = np.exp(-dist_sq / (2.0 * sigma ** 2))
    return W


def compute_knn_affinity(X, k=10, sigma=1.0):
    """
    Vytvoří řídkou k-NN Gaussovskou matici sousednosti:
      - Pro každý bod i najde k nejbližších sousedů
      - Pouze těm přiřadí váhu exp(-dist^2 / (2*sigma^2))
      - Výsledná matice W je pak symetrizovaná: W = max(W, W.T)
    """
    n = X.shape[0]
    W = np.zeros((n, n))

    for i in range(n):
        # Vzdálenosti od bodu i ke všem ostatním
        distances = []
        for j in range(n):
            if i == j:
                continue
            diff = X[i] - X[j]
            dist_sq = np.dot(diff, diff)
            distances.append((dist_sq, j))

        # Seřadíme podle dist_sq a vybereme k nejmenších
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]

        # Přiřadíme Gaussovské váhy
        for dist_sq, j in neighbors:
            W[i, j] = np.exp(-dist_sq / (2.0 * sigma ** 2))

    # Symetrizace (volba max pro nerozšířený k-NN graf)
    W = np.maximum(W, W.T)
    return W


def compute_normalized_laplacian(W):
    """
    Symetrický normalizovaný Laplacian:
       L_sym = D^{-1/2} (D - W) D^{-1/2}.
    Kde D je diagonální matice stupňů (součet řádků W).
    """
    d = np.sum(W, axis=1)  # stupně (degree)
    # Ochrana proti dělení nulou (když by některý vrchol neměl sousedy)
    d_sqrt_inv = 1.0 / np.sqrt(d + 1e-9)
    D_inv_sqrt = np.diag(d_sqrt_inv)

    L = np.diag(d) - W
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    return L_sym


def compute_unnormalized_laplacian(W):
    """
    Nenormalizovaný Laplacian L = D - W.
    """
    d = np.sum(W, axis=1)
    D = np.diag(d)
    L = D - W
    return L


def my_kmeans(X, k, max_iters=100, tol=1e-4):
    """
    Jednoduchá implementace k-means.
    """
    n, d = X.shape
    indices = np.random.choice(n, k, replace=False)
    centers = X[indices, :]
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iters):
        # 1) Přiřazení
        for i in range(n):
            dist_sq = np.sum((X[i] - centers) ** 2, axis=1)
            labels[i] = np.argmin(dist_sq)

        # 2) Přepočítání center
        new_centers = np.zeros((k, d))
        counts = np.zeros(k, dtype=int)
        for i in range(n):
            new_centers[labels[i]] += X[i]
            counts[labels[i]] += 1
        for c in range(k):
            if counts[c] > 0:
                new_centers[c] /= counts[c]
            else:
                # Pokud by některý cluster zůstal prázdný,
                # náhodně ho znovu inicializujeme
                new_centers[c] = X[np.random.randint(0, n)]

        # 3) Kontrola posunu
        shift = np.sum((centers - new_centers) ** 2)
        centers = new_centers
        if shift < tol:
            break

    return labels, centers


def spectral_clustering(
        X,
        k,
        sigma=1.0,
        use_knn=False,
        knn_k=10,
        normalized=True,
        max_iters=100,
        tol=1e-4
):
    """
    Vylepšený Spectral Clustering, který umí:
      - buď plnou Gaussovskou matici sousednosti, nebo k-NN
      - volitelně normalizovaný / nenormalizovaný Laplacian

    Parametry:
      - X: [n, d] vstupní data
      - k: počet shluků
      - sigma: parametr pro Gaussovskou váhu
      - use_knn: pokud True, použije se k-NN graf
      - knn_k: počet sousedů pro k-NN (pokud use_knn=True)
      - normalized: pokud True, použije se symetrický normalizovaný Laplacian
      - max_iters, tol: parametry pro k-means

    Návrat:
      - labels: pole [n], přiřazení clusterů ke každému bodu
    """
    # 1) Afinitní matice
    if use_knn:
        W = compute_knn_affinity(X, k=knn_k, sigma=sigma)
    else:
        W = compute_full_affinity(X, sigma=sigma)

    # 2) Laplacian
    if normalized:
        L = compute_normalized_laplacian(W)
    else:
        L = compute_unnormalized_laplacian(W)

    # 3) Najít k nejmenších vlastních čísel a vektory
    vals, vecs = np.linalg.eigh(L)
    idx_sorted = np.argsort(vals)
    idx_k = idx_sorted[:k]
    U = vecs[:, idx_k]  # [n, k]

    # 4) k-means nad řádky U
    labels, _ = my_kmeans(U, k, max_iters=max_iters, tol=tol)

    return labels
