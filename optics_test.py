#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pro 3D vykreslování v Matplotlibu
import plotly.express as px
from sklearn.datasets import make_moons, make_blobs
import sys, os

# Pokud máte vlastní implementaci v lib/optics.py, odkomentujte a použijte import
# CURR_DIR = os.path.dirname(__file__)
# LIB_DIR = os.path.join(CURR_DIR, 'lib')
# sys.path.append(LIB_DIR)
# from optics import OPTICS

# Pro demonstraci vložíme jednoduchou implementaci přímo sem:
class OPTICS:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.ordering_ = []
        self.reachability_ = None
        self.core_dist_ = None
        self.processed_ = None
        self.labels_ = None

    def fit(self, X):
        import math, heapq
        n_samples = len(X)
        self.reachability_ = [math.inf] * n_samples
        self.core_dist_ = [math.inf] * n_samples
        self.processed_ = [False] * n_samples
        self.ordering_ = []

        # Sousedé (naivně), pro větší dataset by se měl použít space-partitioning
        neighbors_list = []
        for i in range(n_samples):
            neighbors = self._region_query(X, i)
            neighbors_list.append(neighbors)

        # Spočteme core distance
        for i in range(n_samples):
            neighbors = neighbors_list[i]
            if len(neighbors) >= self.min_samples:
                dist_sorted = sorted(self._euclidean(X[i], X[n]) for n in neighbors)
                self.core_dist_[i] = dist_sorted[self.min_samples - 1]

        # Hlavní smyčka
        for i in range(n_samples):
            if not self.processed_[i]:
                self._expand_cluster_order(i, neighbors_list, X)

        return self

    def _expand_cluster_order(self, index, neighbors_list, X):
        import math, heapq

        # Přidáme do ordering
        seeds = []
        self.ordering_.append(index)
        self.processed_[index] = True

        # Pokud je to core point
        if self.core_dist_[index] != math.inf:
            self._update(index, neighbors_list[index], seeds, X)
            while seeds:
                _, next_idx = heapq.heappop(seeds)
                if not self.processed_[next_idx]:
                    self.ordering_.append(next_idx)
                    self.processed_[next_idx] = True
                    if self.core_dist_[next_idx] != math.inf:
                        self._update(next_idx, neighbors_list[next_idx], seeds, X)

    def _update(self, center_idx, neighbors, seeds, X):
        import math, heapq
        for neigh_idx in neighbors:
            if not self.processed_[neigh_idx]:
                new_reach_dist = max(
                    self.core_dist_[center_idx],
                    self._euclidean(X[center_idx], X[neigh_idx])
                )
                if new_reach_dist < self.reachability_[neigh_idx]:
                    self.reachability_[neigh_idx] = new_reach_dist
                    heapq.heappush(seeds, (new_reach_dist, neigh_idx))

    def _region_query(self, X, index):
        return [
            i for i in range(len(X))
            if self._euclidean(X[index], X[i]) <= self.eps
        ]

    def _euclidean(self, a, b):
        import math
        return math.dist(a, b)

    def extract_dbscan(self, eps_dbscan=0.5):
        n_samples = len(self.ordering_)
        self.labels_ = [-1] * n_samples
        cluster_id = 0
        for i in range(n_samples):
            point_idx = self.ordering_[i]
            if self.reachability_[point_idx] > eps_dbscan:
                if self.core_dist_[point_idx] <= eps_dbscan:
                    cluster_id += 1
                    self.labels_[i] = cluster_id
                else:
                    self.labels_[i] = -1
            else:
                self.labels_[i] = cluster_id
        return self.labels_

#
# Pomocná funkce k převodu labelů z "visit order" do původního pořadí bodů
#
def map_labels_to_original(labels_in_order, ordering):
    """
    Vytvoří list labelů ve stejné délce jako je ordering (tedy n_samples).
    labels_in_order[i] je label i-tého bodu v pořadí navštívení.
    ordering[i] je index v původním X.
    Výstup: labels_full, kde labels_full[j] je label j-tého bodu v X.
    """
    n = len(ordering)
    labels_full = [-1] * n
    for visit_i, real_idx in enumerate(ordering):
        labels_full[real_idx] = labels_in_order[visit_i]
    return labels_full


def demo_2d_matplotlib_plot(X, labels_full):
    plt.figure(figsize=(6,5))
    plt.title("OPTICS (2D, Matplotlib)")
    plt.scatter(X[:,0], X[:,1], c=labels_full, cmap='plasma', s=15)
    plt.colorbar(label="Cluster ID")
    plt.show()

def demo_2d_plotly_plot(X, labels_full):
    fig = px.scatter(
        x=X[:,0],
        y=X[:,1],
        color=[str(lbl) for lbl in labels_full],
        title="OPTICS (2D, Plotly)"
    )
    fig.show()

def demo_3d_matplotlib_plot(X, labels_full):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("OPTICS (3D, Matplotlib)")
    sc = ax.scatter(X[:,0], X[:,1], X[:,2], c=labels_full, cmap='plasma', s=15)
    plt.colorbar(sc, label="Cluster ID")
    plt.show()

def demo_3d_plotly_plot(X, labels_full):
    fig_plotly = px.scatter_3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        color=[str(lbl) for lbl in labels_full],
        title="OPTICS (3D, Plotly)"
    )
    fig_plotly.show()

def main():
    # ---------------------------------------------------------
    # A) 2D Demo s "two moons"
    # ---------------------------------------------------------
    X2d, _ = make_moons(n_samples=500, noise=0.05, random_state=42)

    # Spustíme OPTICS
    optics_2d = OPTICS(eps=0.2, min_samples=5).fit(X2d)
    labels_2d_in_order = optics_2d.extract_dbscan(eps_dbscan=0.2)
    labels_2d_full = map_labels_to_original(labels_2d_in_order, optics_2d.ordering_)

    # Matplotlib 2D
    demo_2d_matplotlib_plot(X2d, labels_2d_full)

    # Plotly 2D
    demo_2d_plotly_plot(X2d, labels_2d_full)

    # ---------------------------------------------------------
    # B) 3D Demo s "blobs"
    # ---------------------------------------------------------
    X3d, _ = make_blobs(n_samples=600, centers=4, n_features=3, cluster_std=1.0, random_state=0)

    # Spustíme OPTICS
    optics_3d = OPTICS(eps=2.5, min_samples=5).fit(X3d)
    labels_3d_in_order = optics_3d.extract_dbscan(eps_dbscan=2.5)
    labels_3d_full = map_labels_to_original(labels_3d_in_order, optics_3d.ordering_)

    # Matplotlib 3D
    demo_3d_matplotlib_plot(X3d, labels_3d_full)

    # Plotly 3D
    demo_3d_plotly_plot(X3d, labels_3d_full)

if __name__ == "__main__":
    main()
