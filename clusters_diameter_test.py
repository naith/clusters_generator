import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from lib.cluster_generator import ClusterGenerator


def create_directories(path: str) -> None:
    """Vytvoření adresářové struktury"""
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_poisson_disk_nd(n_dimensions: int = 3,
                             dim_points: int = 100,
                             range_min: float = -20.0,
                             range_max: float = 20.0,
                             fill_percent: float = 0.3,
                             base_max_attempts: int = 200,
                             dtype: str = 'float') -> np.ndarray:
    """
    Generuje n-dimenzionální Poisson disk sampling.
    """
    if n_dimensions < 1:
        raise ValueError("Počet dimenzí musí být alespoň 1")
    if fill_percent <= 0 or fill_percent > 1:
        raise ValueError("fill_percent musí být v rozmezí (0,1]")
    if dtype not in ['float', 'int']:
        raise ValueError("dtype musí být 'float' nebo 'int'")

    max_points = n_dimensions * dim_points
    max_attempts = base_max_attempts * n_dimensions
    bbox_min = np.array([range_min] * n_dimensions, dtype=float)
    bbox_max = np.array([range_max] * n_dimensions, dtype=float)
    diag = np.linalg.norm(bbox_max - bbox_min)
    r = diag * (0.5 - 0.49 * fill_percent)

    points = []
    attempts_count = 0

    while len(points) < max_points and attempts_count < max_attempts:
        if dtype == 'int':
            candidate = np.random.randint(range_min, range_max + 1, size=n_dimensions)
        else:
            candidate = np.random.uniform(bbox_min, bbox_max)

        dists = None

        if len(points) == 0:
            points.append(candidate)
        else:
            dists = np.linalg.norm(np.array(points) - candidate, axis=1)
            if np.all(dists >= r):
                points.append(candidate)

        attempts_count += 1

        if (dists is None) or (len(points) > 0 and np.all(dists >= r)):
            attempts_count = 0

    points = np.array(points)
    if dtype == 'int':
        points = points.astype(int)

    return points


def is_cluster_in_bounds(center: np.ndarray,
                         radius: float,
                         range_min: float,
                         range_max: float) -> bool:
    """
    Kontrola zda je celý cluster v prostoru pro n-dimenzionální prostor
    """
    return np.all((center - radius >= range_min) & (center + radius <= range_max))


def generate_sphere_waypoints(center=None):
    """
    Generování waypoints pro sférický cluster
    """
    if center is None:
        center = [0, 0, 0]
    return np.array([center])


def save_datasets(points_dict: Dict,
                  n_dimensions: int,
                  output_dir: str = 'data/sphere') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Uložení všech datasetů
    """
    create_directories(output_dir)

    dim_cols = [f'dim_{i + 1}' for i in range(n_dimensions)]

    # Background
    background_df = pd.DataFrame(points_dict['background'], columns=dim_cols)
    background_df.to_csv(f'{output_dir}/background_nd.csv', index=False)

    # Clustery
    cluster_data = []
    for points, cluster_id in points_dict['clusters']:
        df = pd.DataFrame(points, columns=dim_cols)
        df['cluster_id'] = f'cluster_{cluster_id}'
        cluster_data.append(df)

    clusters_df = pd.concat(cluster_data) if cluster_data else pd.DataFrame()
    clusters_df.to_csv(f'{output_dir}/clusters_nd.csv', index=False)

    # Kompletní dataset
    background_df['point_type'] = 'background'
    if not clusters_df.empty:
        clusters_df['point_type'] = clusters_df['cluster_id']
        complete_df = pd.concat([background_df, clusters_df])
    else:
        complete_df = background_df

    complete_df.to_csv(f'{output_dir}/sphere_cluster_dataset.csv', index=False)

    return background_df, clusters_df, complete_df


def sphere_clusters_generator():
    """
    Hlavní funkce pro generování sphere clusterů
    """
    fig = go.Figure()

    dd = int(800 / 4)

    rnd_position = [[4, 4 * dd, 'blue'],
                    [6, 2 * dd, 'red'],
                    [3, 3 * dd, 'green'],
                    [12, 12 * dd, 'orange']]

    # Generování Poisson-disk bodů
    fill_percent = 0.9
    poisson_points = generate_poisson_disk_nd(n_dimensions=3, dim_points=666,
                                              range_min=-20, range_max=20,
                                              fill_percent=fill_percent, base_max_attempts=66)

    # Generování pozic pro centra clusterů
    positions = generate_poisson_disk_nd(n_dimensions=1, dim_points=6,
                                         range_min=0, range_max=len(poisson_points),
                                         fill_percent=1, base_max_attempts=66, dtype='int')

    # Pro ukládání dat
    clusters_data = []
    position = 0
    used_indices = set()  # Pro sledování již použitých indexů

    for (radius, density, color) in rnd_position:
        center_found = False
        current_idx = int(positions[position])  # Začínáme na původně vybrané pozici

        # Zkoušíme postupně další body, dokud nenajdeme vhodný nebo nedojdeme na konec
        while current_idx < len(poisson_points) and not center_found:
            if current_idx not in used_indices:
                center = poisson_points[current_idx]

                if is_cluster_in_bounds(center, radius, -20, 20):
                    used_indices.add(current_idx)
                    center_found = True

                    # Cluster sphere
                    waypoints_sphere = generate_sphere_waypoints(center)
                    cg_sphere = ClusterGenerator(waypoints_sphere, tension=0.165,
                                                 cluster_density=density)
                    cg_sphere.generate_shape(tension=0.165, distribution='gauss',
                                             radius=radius)
                    cg_sphere.transform_clusters([30, 30, 0], [1, 1, 1], [0, 0, 0])
                    cg_sphere.add_scatter_traces(fig, color=color, name_prefix='Sphere2',
                                                 show_waypoints=False, show_centers=False,
                                                 show_lines=False)

                    # Uložení bodů clusteru pro export
                    clusters_data.append((cg_sphere.clusters, position))
                else:
                    print(f"Pozice {current_idx} nevyhovuje pro cluster s radiusem {radius}, zkouším další")

            current_idx += 1

        if not center_found:
            print(f"Varování: Nepodařilo se najít vhodné centrum pro cluster s radiusem {radius}")

        position += 1

    # Přidání Poisson-disk bodů do grafu
    fig.add_trace(go.Scatter3d(
        x=poisson_points[:, 0],
        y=poisson_points[:, 1],
        z=poisson_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='gray'),
        name='Poisson-disk (gray)'
    ))

    # Nastavení layoutu
    fig.update_layout(
        title=f'Různé křivky + Poisson-disk  v BOX [-20,20], fill={fill_percent * 100:.0f}%',
        width=1920,
        height=1200,
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[-20, 20], title='X'),
            yaxis=dict(range=[-20, 20], title='Y'),
            zaxis=dict(range=[-20, 20], title='Z')
        )
    )

    # Uložení všech datasetů
    points_dict = {
        'background': poisson_points,
        'clusters': clusters_data
    }
    save_datasets(points_dict, n_dimensions=3)

    fig.show()


if __name__ == "__main__":
    sphere_clusters_generator()