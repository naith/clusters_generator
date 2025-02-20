# lib/optics.py

import math
import heapq


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
        X je numpy-like matice tvaru (n_samples, n_features).
        """
        n_samples = len(X)
        self.reachability_ = [math.inf] * n_samples
        self.core_dist_ = [math.inf] * n_samples
        self.processed_ = [False] * n_samples
        self.ordering_ = []

        # K urychlení můžeme předpočítat seznam sousedů pro každý bod.
        # Nicméně pro velké dataset to může být paměťově náročné.
        # Zde ukázka naivní implementace pro menší dataset.
        neighbors_list = []
        for i in range(n_samples):
            neighbors = self._region_query(X, i)
            neighbors_list.append(neighbors)

        # Pro každý bod určíme jeho core distance
        for i in range(n_samples):
            neighbors = neighbors_list[i]
            if len(neighbors) >= self.min_samples:
                # Seřadíme vzdálenosti od bodu i k sousedům a bereme min_samples-1 vzdálenost
                dist_sorted = sorted([self._euclidean(X[i], X[n]) for n in neighbors])
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
        # Vložíme bod do ordering_ a označíme jej za zpracovaný
        seeds = []
        self.ordering_.append(index)
        self.processed_[index] = True

        # Pokud je bod index 'core point', nastavíme priority queue (seeds)
        if self.core_dist_[index] != math.inf:
            self._update(index, neighbors_list[index], seeds, X)

            # Vybíráme bod s nejmenší reachability vzdáleností z priority queue
            while seeds:
                # seeds je min-heap (díky heapq), proto pop vrací minimum
                _, next_idx = heapq.heappop(seeds)
                if not self.processed_[next_idx]:
                    self.ordering_.append(next_idx)
                    self.processed_[next_idx] = True
                    if self.core_dist_[next_idx] != math.inf:
                        self._update(next_idx, neighbors_list[next_idx], seeds, X)

    def _update(self, center_idx, neighbors, seeds, X):
        """
        Aktualizuje reachability vzdálenosti sousedů a vkládá je do min-heapu (priority queue).
        """
        # Vypočteme tzv. newReachDist = max(core_dist(center_idx), dist(center_idx, neigh)).
        for neigh_idx in neighbors:
            if not self.processed_[neigh_idx]:
                new_reach_dist = max(self.core_dist_[center_idx],
                                     self._euclidean(X[center_idx], X[neigh_idx]))
                if new_reach_dist < self.reachability_[neigh_idx]:
                    self.reachability_[neigh_idx] = new_reach_dist
                    # Přidáme do min-heapu (value = reach_distance, key = neigh_idx)
                    heapq.heappush(seeds, (new_reach_dist, neigh_idx))

    def _region_query(self, X, index):
        """
        Najde a vrátí všechny sousedy bodu 'index' v okruhu eps (včetně bodu samotného).
        """
        neighbors = []
        for i in range(len(X)):
            if self._euclidean(X[index], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _euclidean(self, a, b):
        """
        Euklidovská vzdálenost mezi body a a b (pro n-dim).
        """
        return math.dist(a, b)

    def extract_dbscan(self, eps_dbscan=0.5):
        """
        Extrakce clusterů podobně jako v DBSCANu, s využitím reachability_ a ordering_.
        eps_dbscan: float
            Prahová vzdálenost pro oddělení clusterů v reachability grafu.

        Naplní self.labels_ tak, že pro každý bod přiřadí ID clusteru nebo -1 (noise).
        """
        n_samples = len(self.ordering_)
        self.labels_ = [-1] * n_samples

        cluster_id = 0
        for i in range(n_samples):
            point_idx = self.ordering_[i]

            # Pokud je reachability_ bodu větší než eps_dbscan, vzniká nový cluster (nebo je to šum)
            if self.reachability_[point_idx] > eps_dbscan:
                # Bod je buď začátek nového clusteru, nebo šum
                # Podle core_dist_: pokud je bod core, začínáme nový cluster
                if self.core_dist_[point_idx] <= eps_dbscan:
                    cluster_id += 1
                    self.labels_[point_idx] = cluster_id
                else:
                    # Šum
                    self.labels_[point_idx] = -1
            else:
                # Patří do stávajícího clusteru
                self.labels_[point_idx] = cluster_id

        # self.labels_ je ve smyslu ordering_; pokud potřebujeme ve smyslu původního pořadí indexů,
        # můžeme vrátit namapovanou verzi.
        # Zde však ponecháme takto. Lze případně přemapovat, pokud je potřeba.
        return self.labels_
