import math
import heapq
import numpy as np
from sklearn.neighbors import BallTree

class OPTICS:
    """
    Implementace algoritmu OPTICS (Ordering Points To Identify the Clustering Structure).

    Parametry:
    ----------
    eps : float
        Maximální vzdálenost pro vyhledání sousedů při procházení (tzv. Eps).
    min_samples : int
        Minimální počet bodů, aby byl bod považován za 'core point'.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.ordering_ = []  # Pořadí bodů, v jakém byly navštíveny
        self.reachability_ = None  # Pole s reachability vzdálenostmi
        self.core_dist_ = None  # Pole s core vzdálenostmi
        self.processed_ = None  # Booleovský seznam zpracovaných bodů
        self.labels_ = None  # Vygenerované štítky (cluster ID) po extrakci

    def fit(self, X):
        """
        Aplikuje OPTICS na datovou sadu X.
        """
        n_samples = len(X)
        self.reachability_ = np.full(n_samples, np.inf)  # Inicializace
        self.core_dist_ = np.full(n_samples, np.inf)
        self.processed_ = np.zeros(n_samples, dtype=bool)
        self.ordering_ = []

        # ✅ Použití Ball-Tree pro rychlejší vyhledávání sousedů
        tree = BallTree(X)

        # Předpočítání sousedů
        neighbors_list = tree.query_radius(X, r=self.eps)

        # Výpočet core vzdálenosti pro každý bod
        for i in range(n_samples):
            neighbors = neighbors_list[i]
            if len(neighbors) >= self.min_samples:
                dist_sorted = np.sort(np.linalg.norm(X[neighbors] - X[i], axis=1))
                self.core_dist_[i] = dist_sorted[self.min_samples - 1]

        # Hlavní loop algoritmu
        for i in range(n_samples):
            if not self.processed_[i]:
                self._expand_cluster_order(i, neighbors_list, X)

        return self

    def _expand_cluster_order(self, index, neighbors_list, X):
        """
        Rozšiřuje seřazení clusterů počínaje zadaným bodem.
        """
        seeds = []
        self.ordering_.append(index)
        self.processed_[index] = True

        if self.core_dist_[index] != np.inf:
            self._update(index, neighbors_list[index], seeds, X)

            while seeds:
                _, next_idx = heapq.heappop(seeds)
                if not self.processed_[next_idx]:
                    self.ordering_.append(next_idx)
                    self.processed_[next_idx] = True
                    if self.core_dist_[next_idx] != np.inf:
                        self._update(next_idx, neighbors_list[next_idx], seeds, X)

    def _update(self, center_idx, neighbors, seeds, X):
        """
        Aktualizuje reachability vzdálenosti sousedů a vkládá je do min-heapu (priority queue).
        """
        for neigh_idx in neighbors:
            if not self.processed_[neigh_idx]:
                new_reach_dist = max(self.core_dist_[center_idx],
                                     np.linalg.norm(X[center_idx] - X[neigh_idx]))
                if new_reach_dist < self.reachability_[neigh_idx]:
                    self.reachability_[neigh_idx] = new_reach_dist
                    heapq.heappush(seeds, (new_reach_dist, neigh_idx))

    def extract_dbscan(self, eps_dbscan=None):
        """
        Extrakce clusterů podobně jako v DBSCANu, s využitím reachability_ a ordering_.
        """
        n_samples = len(self.ordering_)
        self.labels_ = np.full(n_samples, -1)

        # ✅ Automatické nastavení eps_dbscan, pokud není zadáno
        if eps_dbscan is None:
            eps_dbscan = np.percentile(self.reachability_[self.reachability_ < np.inf], 90)

        cluster_id = 0
        for i in range(n_samples):
            point_idx = self.ordering_[i]
            if self.reachability_[point_idx] > eps_dbscan:
                if self.core_dist_[point_idx] <= eps_dbscan:
                    cluster_id += 1
                    self.labels_[point_idx] = cluster_id
                else:
                    self.labels_[point_idx] = -1
            else:
                self.labels_[point_idx] = cluster_id

        return self.labels_

    def extract_hierarchical(self, xi=0.05):
        """
        Vytvoří hierarchické clustery na základě změn hustoty.
        """
        cluster_id = 0
        self.labels_ = np.full(len(self.ordering_), -1)
        prev_dist = 0

        for i in range(1, len(self.ordering_)):
            point_idx = self.ordering_[i]
            reach_dist = self.reachability_[point_idx]

            if reach_dist / (prev_dist + 1e-10) > 1 + xi:
                cluster_id += 1
            self.labels_[point_idx] = cluster_id
            prev_dist = reach_dist

        return self.labels_
