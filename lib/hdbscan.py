#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math


def euclidean_distance(a, b):
    """
    Euklidovská vzdálenost mezi dvěma body libovolné dimenze.
    a, b: iterovatelné (x1, x2, ..., xN)
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_core_distances(data, min_points):
    """
    Pro každý bod spočítá 'core distance' = vzdálenost k min_points-tému
    nejbližšímu sousedovi. (Zjednodušená verze HDBSCAN.)

    data: list bodů (každý bod je např. tuple/list souřadnic [x1, x2, ... , xN])
    min_points: např. 5 (podobné parametru min_samples v HDBSCAN)

    Vrací list core_distance pro každý bod.
    """
    n = len(data)
    core_distances = [0.0] * n

    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            d = euclidean_distance(data[i], data[j])
            dists.append(d)
        dists.sort()
        # core distance je vzdálenost k (min_points)-tému nejbližšímu bodu
        # (pozor, index v poli je min_points-1)
        if len(dists) >= min_points:
            core_distances[i] = dists[min_points - 1]
        else:
            # Pokud je bod extrémně izolovaný (méně bodů než min_points),
            # můžeme core_distance nastavit na max nebo něco "velkého".
            core_distances[i] = float('inf')

    return core_distances


def build_mst(data, core_dists):
    """
    Vygeneruje všechny hrany (i,j) a přiřadí jim "reachability distance"
    = max(core_dists[i], d(i,j), core_dists[j]).

    Následně vrací setříděný seznam hran (rdist, i, j).

    data: list bodů
    core_dists: list core vzdáleností
    """
    edges = []
    n = len(data)

    # O(n^2) přístup: pro malé n to postačuje.
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(data[i], data[j])
            rd = max(core_dists[i], d, core_dists[j])
            edges.append((rd, i, j))

    # Seřadit vzestupně podle reachability distance
    edges.sort(key=lambda x: x[0])
    return edges


def kruskal_mst(edges, n):
    """
    Klasický Kruskalův algoritmus pro stavbu minimální kostry (MST).
    edges: setříděné hrany (dist, i, j)
    n: počet vrcholů (bodů)

    Vrací list hran, které tvoří MST (tedy n-1 hran).
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx = find(x)
        ry = find(y)
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


def get_clusters_from_condensed_tree(mst, min_points, min_cluster_size):
    """
    Zjednodušený mechanismus pro zisk klastrů z MST tak, že "odřezáváme" hrany
    od nejdelších k nejkratším (či naopak) a sledujeme komponenty.

    Real HDBSCAN pracuje s "condensed tree" a stabilitou klastrů, tady
    je to velmi zjednodušené kvůli demonstraci.

    Vrací list labelů pro každý bod (0,1,2,...) nebo -1 pro šum.
    """
    if not mst:
        return []

    # z MST zjistíme počet vrcholů
    n = max(max(e[1], e[2]) for e in mst) + 1

    # Seřadíme hrany v MST sestupně (od největší distance k nejmenší)
    sorted_mst = sorted(mst, key=lambda x: x[0], reverse=True)

    # Místo "rozpojování" si vybudujeme union-find od nuly
    # a budeme hrany přidávat od nejmenší k největší (což je
    # opačný postup, ale výsledek je stejný).
    edges_asc = list(reversed(sorted_mst))  # tedy od nejmenší vzdálenosti k největší

    parent2 = list(range(n))
    rank2 = [0] * n

    def find2(x):
        if parent2[x] != x:
            parent2[x] = find2(parent2[x])
        return parent2[x]

    def union2(x, y):
        rx = find2(x)
        ry = find2(y)
        if rx != ry:
            if rank2[rx] < rank2[ry]:
                parent2[rx] = ry
            elif rank2[rx] > rank2[ry]:
                parent2[ry] = rx
            else:
                parent2[ry] = rx
                rank2[rx] += 1

    clusters_final = [None] * n

    # Procházíme hrany od nejmenší k největší, přidáváme je
    # a zjišťujeme, kdy vznikají větší komponenty.
    for dist, i, j in edges_asc:
        union2(i, j)

        # Vytvoříme mapu root -> seznam vrcholů
        comp_map = {}
        for node in range(n):
            r = find2(node)
            comp_map.setdefault(r, []).append(node)

        # Když některá komponenta dosáhne velikosti >= min_cluster_size,
        # přiřadíme jí "cluster ID" (zde root).
        for root, members in comp_map.items():
            if len(members) >= min_cluster_size:
                # label = root (nebo nějaký jiný identifikátor)
                for m in members:
                    if clusters_final[m] is None:
                        clusters_final[m] = root

    # Zbylé body, které nikdy nebyly v dost velké komponentě, označíme -1
    for i in range(n):
        if clusters_final[i] is None:
            clusters_final[i] = -1

    return clusters_final


def hdbscan(data, min_points=5, min_cluster_size=5):
    """
    Zjednodušená (didaktická) verze HDBSCAN pro n-dim data:
      1) Spočítá core distance (vzdálenost k min_points-tému nejbližšímu bodu).
      2) Postaví hrany s reachability distance = max(core_i, d(i,j), core_j).
      3) Zkonstruuje MST (Kruskal).
      4) Ze zjednodušeného "condensed tree" (z MST) vybere klastery
         (podle min_cluster_size).

    Vrací list labelů (0,1,2,...) nebo -1 pro šum.
    """
    # 1) Spočítáme core distance
    core_dists = compute_core_distances(data, min_points)

    # 2) Vygenerujeme hrany a setřídíme (reachability distance)
    edges = build_mst(data, core_dists)

    # 3) Postavíme MST
    mst = kruskal_mst(edges, len(data))

    # 4) Získáme clustery
    labels = get_clusters_from_condensed_tree(mst, min_points, min_cluster_size)

    return labels


# if __name__ == "__main__":
#     # Jednoduchý test pro 2D data
#     data_2d = [
#         (0, 0), (1, 0), (0, 1), (1, 1),  # cluster 1
#         (10, 10), (10, 11), (11, 10),  # cluster 2
#         (50, 50)  # izolovaný bod
#     ]
#     labels_2d = hdbscan(data_2d, min_points=2, min_cluster_size=2)
#     print("Test na 2D datech:", labels_2d)
