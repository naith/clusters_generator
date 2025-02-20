### mean_shift.py
import math
import numpy as np


class MeanShift:
    def __init__(self, bandwidth=1.0, max_iter=300, tol=1e-3):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        points = X.copy()

        for _ in range(self.max_iter):
            new_points = []
            for i in range(n_samples):
                neighbors = self._region_query(points[i], X)
                new_point = np.mean(neighbors, axis=0)
                new_points.append(new_point)
            new_points = np.array(new_points)

            if np.linalg.norm(new_points - points, axis=1).max() < self.tol:
                break
            points = new_points

        self.cluster_centers_ = self._find_unique_centers(points)
        self.labels_ = self._assign_labels(points, self.cluster_centers_)

    def _region_query(self, point, X):
        return np.array([x for x in X if np.linalg.norm(point - x) < self.bandwidth])

    def _find_unique_centers(self, points):
        unique_centers = []
        for p in points:
            if not any(np.linalg.norm(p - c) < self.bandwidth for c in unique_centers):
                unique_centers.append(p)
        return np.array(unique_centers)

    def _assign_labels(self, points, centers):
        labels = np.zeros(len(points), dtype=int)
        for i, p in enumerate(points):
            labels[i] = np.argmin([np.linalg.norm(p - c) for c in centers])
        return labels