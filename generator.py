#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
N-dimenzionální generátor pozadí pomocí modrého šumu s přidanými clustery.
Podporuje generování v libovolném počtu dimenzí.
Vizualizace dostupná pro 2D a 3D data.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, List, Optional, Union
from scipy.spatial import cKDTree


class NDimPoissonDiskSampling:
    """
    N-dimenzionální implementace Poisson Disk Sampling.
    """

    def __init__(self, dims: Union[List[float], Tuple[float, ...]], radius: float, k: int = 20):
        """
        Inicializace generátoru.

        Parameters:
        -----------
        dims : Union[List[float], Tuple[float, ...]]
            Seznam velikostí v každé dimenzi
        radius : float
            Minimální vzdálenost mezi body
        k : int
            Počet pokusů o vygenerování nového bodu
        """
        self.dims = np.array(dims)
        self.n_dims = len(dims)
        self.radius = radius
        self.k = k
        self.cell_size = radius / np.sqrt(self.n_dims)

        self.grid_dims = np.ceil(self.dims / self.cell_size).astype(int)
        self.grid = np.full(self.grid_dims, -1, dtype=int)
        self.samples = []

    def _point_to_grid(self, point: np.ndarray) -> Tuple[int, ...]:
        """Převede souřadnice bodu na indexy v N-dimenzionálním gridu."""
        return tuple((point / self.cell_size).astype(int))

    def _get_grid_neighbors(self, point: np.ndarray) -> List[np.ndarray]:
        """Získá sousední body z N-dimenzionálního gridu."""
        grid_point = self._point_to_grid(point)
        neighbors = []
        r = 2  # Optimalizovaný radius pro prohledávání

        # Vytvoření rozsahů pro každou dimenzi
        ranges = [range(max(0, grid_point[d] - r),
                        min(self.grid_dims[d], grid_point[d] + r + 1))
                  for d in range(self.n_dims)]

        # Iterace přes všechny kombinace indexů
        from itertools import product
        for idx in product(*ranges):
            grid_idx = self.grid[idx]
            if grid_idx != -1:
                neighbors.append(self.samples[grid_idx])

        return neighbors

    def _is_valid_point(self, point: np.ndarray, neighbors: List[np.ndarray]) -> bool:
        """Kontrola platnosti bodu v N-dimenzionálním prostoru."""
        if not np.all((point >= 0) & (point < self.dims)):
            return False

        if not neighbors:
            return True

        for neighbor in neighbors:
            if np.linalg.norm(neighbor - point) < self.radius:
                return False
        return True

    def _generate_random_point_around(self, point: np.ndarray) -> np.ndarray:
        """Generuje náhodný bod v okolí v N-dimenzionálním prostoru."""
        while True:
            # Generování náhodného směru v N-dimenzionálním prostoru
            direction = np.random.normal(0, 1, self.n_dims)
            direction = direction / np.linalg.norm(direction)

            # Generování náhodné vzdálenosti
            r = np.random.uniform(self.radius, 2 * self.radius)

            # Vytvoření nového bodu
            new_point = point + r * direction

            if np.all((new_point >= 0) & (new_point < self.dims)):
                return new_point

    def sample(self) -> np.ndarray:
        """Generuje body pomocí N-dimenzionálního Poisson Disk Sampling."""
        # První bod uprostřed prostoru
        first_point = self.dims / 2
        self.samples.append(first_point)
        grid_point = self._point_to_grid(first_point)
        self.grid[grid_point] = 0

        active_list = [0]

        while active_list:
            idx = np.random.randint(len(active_list))
            point = self.samples[active_list[idx]]
            found_valid = False

            for _ in range(self.k):
                new_point = self._generate_random_point_around(point)
                neighbors = self._get_grid_neighbors(new_point)

                if self._is_valid_point(new_point, neighbors):
                    self.samples.append(new_point)
                    grid_idx = self._point_to_grid(new_point)
                    self.grid[grid_idx] = len(self.samples) - 1
                    active_list.append(len(self.samples) - 1)
                    found_valid = True
                    break

            if not found_valid:
                active_list.pop(idx)

        return np.array(self.samples)


class NDimNoiseVisualizer:
    """Třída pro vizualizaci N-dimenzionálního šumu."""

    @staticmethod
    def visualize(points: np.ndarray,
                  point_types: np.ndarray,
                  title: str = "N-Dimensional Blue Noise with Clusters") -> Optional[go.Figure]:
        """
        Vizualizuje body šumu a clusterů pro 2D a 3D případy.
        Pro vyšší dimenze vrátí None.
        """
        n_dims = points.shape[1]

        if n_dims not in [2, 3]:
            print(f"Vizualizace není dostupná pro {n_dims}D data")
            return None

        # Rozdělení bodů podle typu
        noise_mask = point_types == 0
        cluster_mask = point_types == 1

        if n_dims == 2:
            fig = go.Figure()

            # Přidání šumových bodů
            fig.add_trace(go.Scatter(
                x=points[noise_mask, 0],
                y=points[noise_mask, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color='#808080',
                    opacity=0.6
                ),
                name='Šumové body'
            ))

            # Přidání cluster bodů
            fig.add_trace(go.Scatter(
                x=points[cluster_mask, 0],
                y=points[cluster_mask, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.8
                ),
                name='Cluster body'
            ))

            fig.update_layout(
                title=title,
                xaxis_title='Dimenze 1',
                yaxis_title='Dimenze 2',
                width=800,
                height=800
            )
        else:  # 3D
            fig = go.Figure()

            # Přidání šumových bodů
            fig.add_trace(go.Scatter3d(
                x=points[noise_mask, 0],
                y=points[noise_mask, 1],
                z=points[noise_mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='#808080',
                    opacity=0.6
                ),
                name='Šumové body'
            ))

            # Přidání cluster bodů
            fig.add_trace(go.Scatter3d(
                x=points[cluster_mask, 0],
                y=points[cluster_mask, 1],
                z=points[cluster_mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='blue',
                    opacity=0.8
                ),
                name='Cluster body'
            ))

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Dimenze 1',
                    yaxis_title='Dimenze 2',
                    zaxis_title='Dimenze 3',
                    aspectmode='cube'
                ),
                width=800,
                height=800
            )

        return fig


def generate_cluster_centers(n_dimensions: int = 3, space_size: float = 20.0,
                             initial_radius: float = 5.0, n_centers: int = 10) -> np.ndarray:
    """
    Generuje centra clusterů pomocí existující implementace modrého šumu.
    Automaticky upravuje radius dokud nezíská požadovaný počet bodů.
    """
    dims = [space_size] * n_dimensions
    radius = initial_radius * np.sqrt(n_dimensions)
    max_attempts = 10
    reduction_factor = 0.8

    # Rozdělení bodů do tří zón
    n_edge_points = n_centers // 3
    n_middle_points = n_centers - 2 * n_edge_points

    for attempt in range(max_attempts):
        sampler = NDimPoissonDiskSampling(dims=dims, radius=radius)
        candidates = sampler.sample()

        if len(candidates) >= n_centers * 2:
            # Rozdělení prostoru na zóny
            center = space_size / 2
            inner_radius = space_size / 3

            # Převedeme kandidáty na numpy array
            candidates = np.array(candidates)

            # Výpočet vzdáleností od středu pro všechny body najednou
            distances = np.linalg.norm(candidates - center, axis=1)

            # Rozdělení bodů podle vzdálenosti
            center_mask = distances < inner_radius
            edge_mask = ~center_mask

            center_points = candidates[center_mask]
            edge_points = candidates[edge_mask]

            if len(center_points) >= n_middle_points and len(edge_points) >= 2 * n_edge_points:
                # Náhodný výběr bodů z každé zóny
                edge_indices = np.random.choice(len(edge_points), 2 * n_edge_points, replace=False)
                center_indices = np.random.choice(len(center_points), n_middle_points, replace=False)

                selected_points = np.concatenate([
                    edge_points[edge_indices[:n_edge_points]],
                    center_points[center_indices],
                    edge_points[edge_indices[n_edge_points:]]
                ])

                # Převod do rozsahu [-10, 10]
                selected_points = selected_points - (space_size / 2)
                return selected_points

        print(f"Pokus {attempt + 1}: Radius {radius:.2f} vytvořil {len(candidates)} bodů")
        radius *= reduction_factor

    raise ValueError(f"Nepodařilo se vygenerovat dostatek bodů ani po {max_attempts} pokusech")


def generate_background(n_points: int, n_dimensions: int = 3, space_size: float = 20.0,
                        radius: float = 2.0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generuje N-dimenzionální pozadí s šumovými body a clustery.

    Parameters:
    -----------
    n_points : int
        Požadovaný počet šumových bodů
    n_dimensions : int
        Počet dimenzí (>= 2)
    space_size : float
        Velikost prostoru v každé dimenzi
    radius : float
        Minimální vzdálenost mezi body
    """
    if n_dimensions < 2:
        raise ValueError("Počet dimenzí musí být alespoň 2")

    # Generování šumových bodů
    dims = [space_size] * n_dimensions
    sampler = NDimPoissonDiskSampling(dims=dims, radius=radius)
    noise_points = sampler.sample()

    # Oříznutí na požadovaný počet bodů
    if len(noise_points) > n_points:
        indices = np.random.choice(len(noise_points), n_points, replace=False)
        noise_points = noise_points[indices]

    # Generování bodů pro clustery
    cluster_points = generate_cluster_centers(n_dimensions, space_size)

    # Převod do rozsahu [-10, 10]
    noise_points = noise_points - (space_size / 2)

    # Spojení všech bodů
    all_points = np.vstack([noise_points, cluster_points])
    point_types = np.concatenate([
        np.zeros(len(noise_points)),  # 0 pro šumové body
        np.ones(len(cluster_points))  # 1 pro cluster body
    ])

    # Ověření distribuce
    stats = verify_distribution(all_points, radius)

    return all_points, point_types, stats


def verify_distribution(points: np.ndarray, radius: float) -> dict:
    """Ověří kvalitu distribuce bodů v N-dimenzionálním prostoru."""
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)

    stats = {
        'n_dimensions': points.shape[1],
        'min_distance': np.min(distances[:, 1]),
        'avg_distance': np.mean(distances[:, 1]),
        'max_distance': np.max(distances[:, 1]),
        'std_distance': np.std(distances[:, 1]),
        'target_radius': radius,
        'point_count': len(points)
    }

    return stats


def main():
    """Hlavní funkce pro generování a ukládání datasetů."""
    np.random.seed(42)  # Pro reprodukovatelnost

    total_points = 2000
    dimensions = [2, 3, 4, 5]  # Testované dimenze
    noise_ratios = {
        0.2: int(0.2 * total_points),  # 400 bodů
        0.3: int(0.3 * total_points),  # 600 bodů
        0.4: int(0.4 * total_points)  # 800 bodů
    }

    visualizer = NDimNoiseVisualizer()

    for n_dims in dimensions:
        print(f"\nTestování {n_dims}D prostoru:")

        for ratio, n_points in noise_ratios.items():
            print(f"\nGenerování dat s {ratio * 100}% šumem ({n_points} bodů)...")

            # Generování bodů
            points, point_types, stats = generate_background(
                n_points=n_points,
                n_dimensions=n_dims
            )

            # Výpis statistik
            print("\nStatistiky distribuce:")
            print(f"- Počet dimenzí: {stats['n_dimensions']}")
            print(f"- Minimální vzdálenost: {stats['min_distance']:.3f}")
            print(f"- Průměrná vzdálenost: {stats['avg_distance']:.3f}")
            print(f"- Maximální vzdálenost: {stats['max_distance']:.3f}")
            print(f"- Směrodatná odchylka: {stats['target_radius']}")
            print(f"- Celkový počet bodů: {stats['point_count']}")
            print(f"- Počet šumových bodů: {np.sum(point_types == 0)}")
            print(f"- Počet cluster bodů: {np.sum(point_types == 1)}")

            # Uložení do CSV
            df = pd.DataFrame(
                points,
                columns=[f'dim_{i + 1}' for i in range(n_dims)]
            )
            df['typ'] = ['šum' if t == 0 else 'cluster' for t in point_types]

            filename = f'background_noise_{n_dims}d_{int(ratio * 100)}percent_with_clusters.csv'
            df.to_csv(f"data/generator/{filename}", index=False)
            print(f"Body uloženy do: {filename}")

            # Vizualizace (pouze pro 2D a 3D)
            if n_dims in [2, 3]:
                fig = visualizer.visualize(
                    points,
                    point_types,
                    title=f"{n_dims}D Data - {int(ratio * 100)}% šum + {n_points} clusterů"
                )
                if fig:
                    fig.show()


if __name__ == "__main__":
    main()