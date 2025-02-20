class DBSCAN:
    def __init__(self, eps, min_samples):
        """
        Inicializace DBSCAN algoritmu.

        Args:
            eps: Maximální vzdálenost mezi dvěma body pro jejich považování za sousedy
            min_samples: Minimální počet bodů potřebných k vytvoření clusteru
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _euclidean_distance(self, point1, point2):
        """
        Výpočet Euklidovské vzdálenosti mezi dvěma body.

        Args:
            point1: První bod
            point2: Druhý bod

        Returns:
            float: Vzdálenost mezi body
        """
        squared_dist = 0
        for i in range(len(point1)):
            squared_dist += (point1[i] - point2[i]) ** 2
        return squared_dist ** 0.5

    def _region_query(self, X, point_idx):
        """
        Nalezení všech bodů v okolí daného bodu.

        Args:
            X: Dataset
            point_idx: Index zkoumaného bodu

        Returns:
            list: Seznam indexů sousedních bodů
        """
        neighbors = []
        for i in range(len(X)):
            if self._euclidean_distance(X[point_idx], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """
        Rozšíření clusteru o další body.

        Args:
            X: Dataset
            labels: Pole labelů
            point_idx: Index aktuálního bodu
            neighbors: Seznam sousedů
            cluster_id: ID aktuálního clusteru
        """
        labels[point_idx] = cluster_id
        queue = list(neighbors)

        while queue:
            neighbor_idx = queue.pop(0)
            if labels[neighbor_idx] == -1:  # noise
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0:  # unvisited
                labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)

    def _k_distance(self, X, k):
        """
        Výpočet k-distance pro všechny body.

        Args:
            X: Dataset
            k: Počet sousedů

        Returns:
            list: Seřazené k-distance pro všechny body
        """
        distances = []
        for i in range(len(X)):
            dist_to_point = []
            for j in range(len(X)):
                if i != j:
                    dist = self._euclidean_distance(X[i], X[j])
                    dist_to_point.append(dist)
            dist_to_point.sort()
            distances.append(dist_to_point[k - 1])
        distances.sort()
        return distances

    def fit(self, X):
        """
        Provedení DBSCAN clustering algoritmu.

        Args:
            X: Dataset pro clustering

        Returns:
            array: Pole labelů clusterů
        """
        self.labels_ = [0] * len(X)  # 0 = unvisited, -1 = noise
        cluster_id = 0

        for point_idx in range(len(X)):
            if self.labels_[point_idx] != 0:  # skip visited points
                continue

            neighbors = self._region_query(X, point_idx)

            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1  # mark as noise
            else:
                cluster_id += 1
                self._expand_cluster(X, self.labels_, point_idx, neighbors, cluster_id)

        return self.labels_
