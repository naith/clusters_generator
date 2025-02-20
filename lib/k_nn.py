### knn.py
### knn.py
import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def predict(self, X_train, y_train, X):
        X = np.asarray(X)
        predictions = [self._predict(X_train, y_train, x) for x in X]
        return np.array(predictions)

    def _predict(self, X_train, y_train, x):
        distances = [np.linalg.norm(x - x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]