import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from lib.cluster_generator import ClusterGenerator


def create_directories(path: str) -> None:
    """Creates directory structure"""
    Path(path).mkdir(parents=True, exist_ok=True)


def check_cluster_distance(center: np.ndarray,
                           radius: float,
                           existing_clusters: List[Tuple[np.ndarray, float]],
                           min_distance_factor: float = 0.75) -> bool:
    """
    Checks minimum distance between clusters

    Parameters:
    -----------
    center : np.ndarray
        Center of new cluster
    radius : float
        Radius of new cluster
    existing_clusters : List[Tuple[np.ndarray, float]]
        List of existing clusters in format [(center, radius), ...]
    min_distance_factor : float
        Minimum required distance as multiple of radii sum

    Returns:
    --------
    bool
        True if distance is valid, False otherwise
    """
    if not existing_clusters:
        return True

    for exist_center, exist_radius in existing_clusters:
        distance = np.linalg.norm(center - exist_center)
        min_required_distance = (radius + exist_radius) * min_distance_factor

        if distance < min_required_distance:
            return False

    return True


def generate_poisson_disk_nd(n_dimensions: int = 3,
                             dim_points: int = 100,
                             range_min: float = -20.0,
                             range_max: float = 20.0,
                             fill_percent: float = 0.3,
                             base_max_attempts: int = 200,
                             dtype: str = 'float') -> np.ndarray:
    """
    Generates n-dimensional Poisson disk sampling

    Parameters:
    -----------
    n_dimensions : int
        Number of dimensions (default 3)
    dim_points : int
        Points per dimension (default 100)
    range_min : float
        Minimum range value (default -20.0)
    range_max : float
        Maximum range value (default 20.0)
    fill_percent : float
        Fill percentage (0-1) - affects point density
    base_max_attempts : int
        Base number of attempts for point placement
    dtype : str
        Data type ('float' or 'int')

    Returns:
    --------
    np.ndarray
        Generated points array
    """
    if n_dimensions < 1:
        raise ValueError("Number of dimensions must be at least 1")
    if fill_percent <= 0 or fill_percent > 1:
        raise ValueError("fill_percent must be in range (0,1]")
    if dtype not in ['float', 'int']:
        raise ValueError("dtype must be 'float' or 'int'")

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
    Checks if cluster is fully within space bounds
    """
    return np.all((center - radius >= range_min) & (center + radius <= range_max))


def generate_sphere_waypoints(center=None):
    """
    Generates waypoints for spherical cluster
    """
    if center is None:
        center = [0, 0, 0]
    return np.array([center])


def save_datasets(points_dict: Dict,
                  n_dimensions: int,
                  output_dir: str = 'data/sphere') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Saves all datasets

    Parameters:
    -----------
    points_dict : Dict
        Dictionary containing points data
    n_dimensions : int
        Number of dimensions
    output_dir : str
        Output directory path

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Background, clusters and complete dataframes
    """
    create_directories(output_dir)

    dim_cols = [f'dim_{i + 1}' for i in range(n_dimensions)]

    # Background
    background_df = pd.DataFrame(points_dict['background'], columns=dim_cols)
    background_df.to_csv(f'{output_dir}/background_nd.csv', index=False)

    # Clusters
    cluster_data = []
    for points, cluster_id in points_dict['clusters']:
        df = pd.DataFrame(points, columns=dim_cols)
        df['cluster_id'] = f'cluster_{cluster_id}'
        cluster_data.append(df)

    clusters_df = pd.concat(cluster_data) if cluster_data else pd.DataFrame()
    clusters_df.to_csv(f'{output_dir}/clusters_nd.csv', index=False)

    # Complete dataset
    background_df['point_type'] = 'background'
    if not clusters_df.empty:
        clusters_df['point_type'] = clusters_df['cluster_id']
        complete_df = pd.concat([background_df, clusters_df])
    else:
        complete_df = background_df

    complete_df.to_csv(f'{output_dir}/sphere_cluster_dataset.csv', index=False)

    return background_df, clusters_df, complete_df


def visualize_config(fig, poisson_points, fill_percent):
    fig.add_trace(go.Scatter3d(
        x=poisson_points[:, 0],
        y=poisson_points[:, 1],
        z=poisson_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='gray'),
        name='Poisson-disk-noise (gray)'
    ))

    fig.update_layout(
        # Zmenšíme okraje kolem celé plochy
        margin=dict(l=20, r=20, t=20, b=20),

        # Titulek grafu
        # title=f'Various shapes + Poisson-disk in BOX [-20,20], fill={fill_percent * 100:.0f}%',
        title=dict(
            text='Sphere clusters + Poisson-disk noise in BOX [-20,20]',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font = dict(size=20)
        ),

        # Velikost výstupu
        width=1200,
        height=900,

        legend=dict(
            font=dict(size=16),
            x=1.0,
            y=0.0,
            xanchor='right',
            yanchor='bottom',
        ),

        # Nastavení 3D scény
        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),

            # Nastavení os X, Y, Z
            xaxis=dict(
                range=[-20, 20],
                title='X',
                titlefont=dict(size=16),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                range=[-20, 20],
                title='Y',
                titlefont=dict(size=16),
                tickfont=dict(size=14),
            ),
            zaxis=dict(
                range=[-20, 20],
                title='Z',
                titlefont=dict(size=16),
                tickfont=dict(size=14),
            )
        )
    )


def sphere_clusters_generator():
    """
    Main function for generating sphere clusters
    """
    fig = go.Figure()

    dd = int(800 / 4)

    rnd_position = [[4, 4 * dd, 'blue'],
                    [6, 6 * dd, 'red'],
                    [2, 6 * dd, 'brown'],
                    [3, 3 * dd, 'green'],
                    [5, 5 * dd, 'magenta'],
                    [12, 12 * dd, 'orange']]

    # Generate Poisson-disk points
    fill_percent = 0.9
    poisson_points = generate_poisson_disk_nd(
        n_dimensions=3, dim_points=666,
        range_min=-20, range_max=20,
        fill_percent=fill_percent, base_max_attempts=66)

    # Generate positions for cluster centers
    positions = generate_poisson_disk_nd(
        n_dimensions=1, dim_points=6,
        range_min=0, range_max=len(poisson_points),
        fill_percent=1, base_max_attempts=66, dtype='int')

    # For data storage
    clusters_data = []
    position = 0
    used_indices = set()
    existing_clusters = []  # List for distance checking [(center, radius), ...]

    for (radius, density, color) in rnd_position:
        center_found = False
        current_idx = int(positions[position])

        while current_idx < len(poisson_points) and not center_found:
            if current_idx not in used_indices:
                center = poisson_points[current_idx]

                if (is_cluster_in_bounds(center, radius, -20, 20) and
                        check_cluster_distance(center, radius, existing_clusters)):
                    used_indices.add(current_idx)
                    center_found = True

                    # Add cluster to list for distance checking
                    existing_clusters.append((center, radius))

                    # Create sphere cluster
                    waypoints_sphere = generate_sphere_waypoints(center)
                    cg_sphere = ClusterGenerator(
                        waypoints_sphere, tension=0.165,
                        cluster_density=density)

                    cg_sphere.generate_shape(
                        tension=0.165, distribution='gauss', radius=radius)

                    cg_sphere.transform_clusters([30, 30, 0], [1, 1, 1], [0, 0, 0])

                    cg_sphere.add_scatter_traces(
                        fig, color=color, name_prefix='Sphere2',
                        show_waypoints=False, show_centers=False,
                        show_lines=False)

                    clusters_data.append((cg_sphere.clusters, position))
                else:
                    if not is_cluster_in_bounds(center, radius, -20, 20):
                        print(f"Position {current_idx} invalid - cluster would exceed boundaries")
                    else:
                        print(f"Position {current_idx} invalid - too close to existing clusters")

            current_idx += 1

        if not center_found:
            print(f"Warning: Could not find suitable center for cluster with radius {radius}")

        position += 1

    visualize_config(fig=fig, poisson_points=poisson_points, fill_percent=fill_percent)

    # Save all datasets
    points_dict = {
        'background': poisson_points,
        'clusters': clusters_data
    }

    save_datasets(points_dict, n_dimensions=3)

    fig.show()


if __name__ == "__main__":
    sphere_clusters_generator()
