#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generátor N-dimenzionálních dat různých typů:
- Pozadí (blue noise)
- Clustery
- Vlákna
- N-dimenzionální povrchy
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List, Optional, Union, Dict
from scipy.spatial import cKDTree
from itertools import product


class NDimPoissonDiskSampling:

    def __init__(self, dims: Union[List[float], Tuple[float, ...]], radius: float, k: int = 20):
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


class NDimSphereClusterGenerator:

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def _generate_uniform_points(
            self,
            n_points: int,
            n_dimensions: int,
            radius: float
    ) -> np.ndarray:

        points = []
        attempts = 0
        max_attempts = n_points * 10  # Maximální počet pokusů

        while len(points) < n_points and attempts < max_attempts:
            # Generování bodů v n-dimenzionální krychli
            point = np.random.uniform(-radius, radius, n_dimensions)

            # Test, zda bod leží v kouli
            if np.sum(point ** 2) <= radius ** 2:
                points.append(point)

            attempts += 1

        if len(points) < n_points:
            print(f"Varování: Vygenerováno pouze {len(points)} z {n_points} požadovaných bodů")

        return np.array(points)


    def _generate_gaussian_points(
            self,
            n_points: int,
            n_dimensions: int,
            radius: float
    ) -> np.ndarray:

        # Generování pomocí normálního rozdělení
        points = np.random.normal(0, radius / 3, (n_points, n_dimensions))

        # Výpočet vzdáleností od středu
        distances = np.sqrt(np.sum(points ** 2, axis=1))

        # Oříznutí bodů na kouli
        points[distances > radius] *= radius / distances[distances > radius, np.newaxis]

        return points

    def generate_sphere_cluster(
            self,
            n_points: int,
            n_dimensions: int,
            center: Union[List[float], Tuple[float, ...]],
            radius: float,
            distribution: str = 'uniform',
            noise_scale: float = 0.1
    ) -> pd.DataFrame:

        if n_dimensions < 2:
            raise ValueError("Počet dimenzí musí být alespoň 2")

        if len(center) != n_dimensions:
            raise ValueError("Délka středu musí odpovídat počtu dimenzí")

        # Generování bodů podle zvoleného rozložení
        if distribution == 'uniform':
            points = self._generate_uniform_points(n_points, n_dimensions, radius)
        else:  # gaussian
            points = self._generate_gaussian_points(n_points, n_dimensions, radius)

        # Přidání šumu
        if noise_scale > 0:
            noise = np.random.normal(0, radius * noise_scale, (n_points, n_dimensions))
            points += noise

        # Posunutí do středu clusteru
        points += np.array(center)

        # Vytvoření DataFrame
        columns = [f'dim_{i + 1}' for i in range(n_dimensions)]
        df = pd.DataFrame(points, columns=columns)
        df['cluster_type'] = 'sphere'

        return df


class DataGenerator:

    def __init__(self, random_seed: int = 42):

        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate_background(self, n_points: int, n_dimensions: int = 3,
                            space_size: float = 20.0, radius: float = 2.0) -> Tuple[np.ndarray, dict]:

        if n_dimensions < 2:
            raise ValueError("Počet dimenzí musí být alespoň 2")

        dims = [space_size] * n_dimensions
        sampler = NDimPoissonDiskSampling(dims=dims, radius=radius)
        points = sampler.sample()

        # Oříznutí na požadovaný počet bodů
        if len(points) > n_points:
            indices = np.random.choice(len(points), n_points, replace=False)
            points = points[indices]

        # Převod do rozsahu [-10, 10]
        points = points - (space_size / 2)

        # Ověření distribuce
        stats = self._verify_distribution(points, radius)

        return points, stats

    def generate_clusters_centers(self, n_centers: int, n_dimensions: int, radius: float) -> np.ndarray:
        # Vygenerujeme více bodů pomocí blue noise, abychom měli z čeho vybírat

        pass

    def generate_waypoints(self, n_points: int, n_dimensions: int) -> np.ndarray:
        # TODO: Implementace
        pass

    def generate_clusters(self, centers: np.ndarray, n_points_per_cluster: int,
                          radius: float) -> np.ndarray:
        # TODO: Implementace
        pass

    def generate_strings(self, waypoints: np.ndarray, n_points_per_string: int) -> np.ndarray:
        # TODO: Implementace
        pass

    def generate_hypersurface(self, n_points: int, n_dimensions: int) -> np.ndarray:
        # TODO: Implementace
        pass

    def _verify_distribution(self, points: np.ndarray, radius: float) -> dict:

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

    def generate_sphere_cluster(
            self,
            n_points: int,
            center: Tuple[float, float, float],
            radius: float,
            distribution: str = 'uniform',
            noise_scale: float = 0.1
    ) -> pd.DataFrame:
        # Generování náhodných úhlů
        phi = np.random.uniform(0, 2 * np.pi, n_points)  # azimut
        theta = np.arccos(2 * np.random.uniform(0, 1, n_points) - 1)  # elevace

        # Generování vzdáleností od středu podle zvoleného rozložení
        if distribution == 'uniform':
            r = radius * np.cbrt(np.random.uniform(0, 1, n_points))
        else:  # gaussian
            r = np.abs(np.random.normal(0, radius / 3, n_points))
            r = np.clip(r, 0, radius)

        # Převod do kartézských souřadnic
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        # Přidání gaussovského šumu
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, (n_points, 3))
            x += noise[:, 0]
            y += noise[:, 1]
            z += noise[:, 2]

        # Posunutí do středu clusteru
        x += center[0]
        y += center[1]
        z += center[2]

        # Vytvoření DataFrame
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'cluster_type': 'sphere'
        })

        return df


def merge_datasets(data_dict: Dict[str, np.ndarray], n_dimensions: int) -> pd.DataFrame:
    merged_data = []

    # Dimenze pro všechny body
    dimension_cols = [f'dim_{i + 1}' for i in range(n_dimensions)]

    # Zpracování jednotlivých typů dat
    for data_type, points in data_dict.items():
        if points is not None and len(points) > 0:
            # Vytvoření DataFrame pro aktuální typ
            df = pd.DataFrame(points, columns=dimension_cols)
            # Přidání typu bodu
            df['type'] = data_type
            merged_data.append(df)

    # Sloučení všech dataframů
    if merged_data:
        return pd.concat(merged_data, ignore_index=True)
    else:
        # Vytvoření prázdného DataFrame se správnou strukturou
        return pd.DataFrame(columns=dimension_cols + ['type'])


def plot_data_2_3d(colors, merged_dataframe, n_dimensions):
    # Vytvoření scatter plotu s definovanou velikostí
    fig = None
    if (n_dimensions == 2):
        fig = px.scatter(
            merged_dataframe,
            x='dim_1',
            y='dim_2',
            color='type',  # Barevná škála podle 'type'
            color_discrete_map=colors,  # Mapování vlastních barev
            labels={'dim_1': 'Dimension 1', 'dim_2': 'Dimension 2'},  # Popis os
            title='Scatter plot podle typu',
            width=1000,  # Šířka grafu
            height=800  # Výška grafu

        )
        fig.update_layout(
            xaxis=dict(range=[-12, 12]),  # Rozsah osy X
            yaxis=dict(range=[-12, 12])  # Rozsah osy Y
        )

    elif (n_dimensions == 3):
        fig = px.scatter_3d(
            merged_dataframe,
            x='dim_1',
            y='dim_2',
            z='dim_3',
            color='type',  # Barevná škála podle 'type'
            color_discrete_map=colors,  # Mapování vlastních barev
            labels={'dim_1': 'Dimension 1', 'dim_2': 'Dimension 2', 'dim_3': 'Dimension 3'},  # Popis os
            title='Scatter plot podle typu',
            width=1000,  # Šířka grafu
            height=800  # Výška grafu
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-12, 12]),  # Rozsah osy X
                yaxis=dict(range=[-12, 12]),  # Rozsah osy Y
                zaxis=dict(range=[-12, 12])  # Rozsah osy Z
            )
        )
    else:
        print("Unsupported number of dimensions")

    # Zobrazení grafu
    if fig:
        fig.show()


def main():
    """
    Ukázka použití generátoru pro různé typy dat a dimenze.
    """
    # Inicializace generátoru
    generator = DataGenerator(random_seed=42)

    # Definice parametrů pro testování
    dimensions = [2, 3]  # Testované dimenze
    total_points = 2000  # Celkový počet bodů
    noise_ratios = {
        0.2: int(0.2 * total_points),  # 400 bodů
        0.3: int(0.3 * total_points),  # 600 bodů
        0.4: int(0.4 * total_points)  # 800 bodů
    }

    space_size = 20.0
    radius = 2.0

    for n_dimensions in dimensions:
        merged_dataset = []
        print(f"\n{'=' * 50}")
        points, stats = generator.generate_background(
            n_points=int(0.2 * total_points),
            n_dimensions=n_dimensions,
            space_size=space_size,
            radius=radius
        )

        column_names = [f'dim_{i + 1}' for i in range(n_dimensions)]
        df = pd.DataFrame(points, columns=column_names)

        # test background
        df['type'] = 'background'

        df.at[25, 'type'] = 'cluster'
        df.at[4, 'type'] = 'cluster'
        df.at[64, 'type'] = 'cluster'
        df.at[34, 'type'] = 'cluster'

        # Definuj barvy pro jednotlivé typy
        colors = {
            'background': 'grey',
            'cluster': 'blue'
        }

        # Přidání barev do DataFrame podle typu
        df['color'] = df['type'].map(colors)
        print(df)
        plot_data_2_3d(colors, df, n_dimensions)


def create_output_directories():
    """Vytvoření adresářů pro výstupní soubory."""
    import os

    directories = ['data']

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Vytvořen adresář: {directory}")


if __name__ == "__main__":
    create_output_directories()
    main()

# merged_dataset.append(df)

# test clusters
# prototype for merging data
# df1 = df.copy()
# df1['type'] = 'cluster'
# merged_dataset.append(df1)
#
# merged_dataframe = pd.concat(merged_dataset, ignore_index=True)
#
# filename = f'data/complete_dataset_{n_dimensions}d_{int(0.2 * 100)}percent.csv'
#
# merged_dataframe.to_csv(filename, index=False)
#
# print(f"\nSloučený dataset uložen do: {filename}")
#
# fig = generator.visualize_data(
#     points,
#     "test",
#     title=f"{n_dimensions}D test - {int(0.2 * 100)}% bodů"
# )
#
# if fig:
#     fig.show()
