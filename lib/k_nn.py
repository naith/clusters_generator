### knn.py
### knn.py
import numpy as np
from collections import Counter

import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def predict(self, X_train, y_train, X):
        X = np.asarray(X)
        if len(X_train) < self.k:
            raise ValueError("k nesmí být větší než počet vzorků v trénovací sadě.")

        return np.array([self._predict(X_train, y_train, x) for x in X])

    def _predict(self, X_train, y_train, x):
        distances = np.linalg.norm(X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        weights = 1 / (distances[k_indices] + 1e-5)  # Přidán offset pro stabilitu

        label_weights = {}
        for label, weight in zip(k_nearest_labels, weights):
            label_weights[label] = label_weights.get(label, 0) + weight

        return max(label_weights, key=label_weights.get)  # Vybere třídu s nejvyšším váženým skóre

