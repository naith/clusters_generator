import numpy as np
from sklearn.neighbors import BallTree

class DBSCAN:
    def __init__(self, eps=None, min_samples=5, auto_eps=False, k_distance_k=5):
        """
        Inicializace DBSCAN algoritmu.

        Args:
            eps: Maximální vzdálenost mezi dvěma body pro jejich považování za sousedy
            min_samples: Minimální počet bodů potřebných k vytvoření clusteru
            auto_eps: Automatický výpočet `eps` pomocí k-distance heuristiky
            k_distance_k: Počet sousedů pro k-distance (pokud `auto_eps=True`)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.auto_eps = auto_eps
        self.k_distance_k = k_distance_k
        self.labels_ = None

    def _region_query(self, X, tree, point_idx):
        """
        Najde všechny sousedy bodu v okruhu `eps` pomocí BallTree.

        Args:
            X: Dataset
            tree: BallTree instance pro efektivní vyhledávání
            point_idx: Index bodu

        Returns:
            list: Indexy sousedních bodů
        """
        return tree.query_radius(X[point_idx].reshape(1, -1), r=self.eps)[0].tolist()

    def _expand_cluster(self, X, tree, labels, point_idx, neighbors, cluster_id):
        """
        Rozšíří cluster o nové body.

        Args:
            X: Dataset
            tree: BallTree instance
            labels: Pole labelů
            point_idx: Index bodu
            neighbors: Seznam sousedů
            cluster_id: ID clusteru
        """
        labels[point_idx] = cluster_id
        queue = set(neighbors)  # Používáme množinu pro efektivitu

        while queue:
            neighbor_idx = queue.pop()
            if labels[neighbor_idx] == -1:  # noise → přidat do clusteru
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0:  # unvisited → expandovat
                labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, tree, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    queue.update(new_neighbors)  # Přidáme nové sousedy

    def _k_distance(self, X, k):
        """
        Automatický výpočet `eps` pomocí k-distance heuristiky.

        Args:
            X: Dataset
            k: Počet sousedů

        Returns:
            float: Doporučené `eps`
        """
        tree = BallTree(X)
        distances, _ = tree.query(X, k=k + 1)  # k+1, protože první soused je bod samotný
        k_distances = np.sort(distances[:, -1])  # k-tá nejbližší vzdálenost

        # Automatická volba eps na základě největšího skoku (koleno křivky)
        diffs = np.diff(k_distances)
        eps_auto = k_distances[np.argmax(diffs) + 1]
        return eps_auto

    def fit(self, X):
        """
        Provede DBSCAN clustering.

        Args:
            X: Dataset

        Returns:
            list: Labely clusterů pro každý bod
        """
        n = len(X)
        self.labels_ = np.zeros(n, dtype=int)  # 0 = unvisited, -1 = noise
        tree = BallTree(X)

        # Automatický výpočet `eps`, pokud není zadáno
        if self.auto_eps or self.eps is None:
            self.eps = self._k_distance(X, self.k_distance_k)

        cluster_id = 0

        for point_idx in range(n):
            if self.labels_[point_idx] != 0:  # skip visited points
                continue

            neighbors = self._region_query(X, tree, point_idx)

            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1  # označení jako šum
            else:
                cluster_id += 1
                self._expand_cluster(X, tree, self.labels_, point_idx, neighbors, cluster_id)

        return self.labels_