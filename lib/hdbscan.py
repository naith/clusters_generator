import numpy as np
from sklearn.neighbors import BallTree
import heapq

def compute_core_distances(data, min_points):
    """
    Spočítá core distance pro každý bod pomocí k-nejbližších sousedů.
    Používá BallTree pro efektivní dotazy (rychlejší než O(n²)).
    """
    tree = BallTree(data)
    dists, _ = tree.query(data, k=min_points + 1)  # Min_points + 1, protože první je bod samotný
    return dists[:, -1]  # Min_points-tý nejbližší soused

def build_knn_edges(data, core_dists, k=10):
    """
    Vytvoří k-nejbližší sousedy graf pro rychlejší běh HDBSCAN.
    """
    tree = BallTree(data)
    dists, indices = tree.query(data, k=k + 1)  # První bod je sám sebe, ignorujeme ho
    edges = []
    for i in range(len(data)):
        for j in range(1, k + 1):  # Ignorujeme první (vlastní bod)
            rd = max(core_dists[i], dists[i, j], core_dists[indices[i, j]])
            edges.append((rd, i, indices[i, j]))
    edges.sort(key=lambda x: x[0])  # Seřadíme podle reachability distance
    return edges

def kruskal_mst(edges, n):
    """
    Kruskalův algoritmus na vytvoření minimálního spojovacího stromu (MST).
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True
        return False

    mst = []
    for dist, i, j in edges:
        if union(i, j):
            mst.append((dist, i, j))
        if len(mst) == n - 1:
            break
    return mst

def get_clusters_from_mst(mst, min_cluster_size):
    """
    Extrahuje clustery z MST ořezáváním nejdelších hran.
    """
    n = max(max(e[1], e[2]) for e in mst) + 1
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
            return True
        return False

    # Seřazení MST sestupně podle vzdálenosti
    sorted_mst = sorted(mst, key=lambda x: x[0], reverse=True)

    # Inicializace labelů
    labels = [-1] * n

    # Průchod MST a postupné "řezání" hran
    cluster_id = 0
    for dist, i, j in sorted_mst:
        if dist > np.percentile([e[0] for e in sorted_mst], 90):  # Řežeme nejdelší 10% hran
            continue
        union(i, j)

    # Mapování bodů na jejich komponenty
    cluster_map = {}
    for i in range(n):
        root = find(i)
        cluster_map.setdefault(root, []).append(i)

    # Pouze velké komponenty jsou clustery
    for root, members in cluster_map.items():
        if len(members) >= min_cluster_size:
            for m in members:
                labels[m] = cluster_id
            cluster_id += 1

    return labels

def hdbscan(data, min_points=5, min_cluster_size=5, k=10):
    """
    Zlepšená verze HDBSCAN:
      1) Používá BallTree pro výpočet core distances.
      2) Staví k-nejbližší sousedy graf místo plného grafu.
      3) Používá Kruskalův algoritmus k vytvoření MST.
      4) Extrahuje clustery z MST ořezáním nejdelších hran.
    """
    # 1) Spočítáme core distance
    core_dists = compute_core_distances(data, min_points)

    # 2) Vygenerujeme graf k-nejbližších sousedů
    edges = build_knn_edges(data, core_dists, k=k)

    # 3) Postavíme MST
    mst = kruskal_mst(edges, len(data))

    # 4) Extrahujeme clustery
    labels = get_clusters_from_mst(mst, min_cluster_size)

    return labels
