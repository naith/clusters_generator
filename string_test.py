import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.interpolate import CubicSpline


def create_directories(path: str) -> None:
    """Creates directory structure"""
    Path(path).mkdir(parents=True, exist_ok=True)


class ClusterGenerator:
    """
    Třída pro generování a správu clusterů
    """

    def __init__(self, waypoints: np.ndarray, tension: float = 0.5, cluster_density: int = 100):
        """
        Inicializace generátoru clusterů

        Parameters:
        -----------
        waypoints : np.ndarray
            Body definující tvar (kontrolní body)
        tension : float
            Napětí křivky (0-1)
        cluster_density : int
            Hustota bodů v clusteru
        """
        self.waypoints = waypoints
        self.tension = tension
        self.cluster_density = cluster_density
        self.clusters = None
        self.transformed_points = None
        self.line_points = None

    def generate_shape(self, tension: float, distribution: str = 'uniform', radius: float = 1.0):
        """
        Generuje body clusteru podél křivky
        """
        # Interpolace bodů podél křivky
        t = np.linspace(0, 1, self.cluster_density)
        cs = CubicSpline(np.linspace(0, 1, len(self.waypoints)), self.waypoints, axis=0)
        self.line_points = cs(t)

        # Generování bodů clusteru
        clusters = []
        for point in self.line_points:
            if distribution == 'uniform':
                # Uniformní distribuce v kouli
                phi = np.random.uniform(0, 2 * np.pi)
                cos_theta = np.random.uniform(-1, 1)
                theta = np.arccos(cos_theta)
                r = radius * np.random.uniform(0, 1) ** (1 / 3)

                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)

                cluster_point = point + np.array([x, y, z])

            else:  # gaussian
                # Gaussovská distribuce
                noise = np.random.normal(0, radius / 3, 3)
                cluster_point = point + noise

            clusters.append(cluster_point)

        self.clusters = np.array(clusters)
        self.transformed_points = self.clusters.copy()

    def transform_clusters(self, rotation_angles: np.ndarray, scale: List[float], translation: List[float]):
        """
        Aplikuje transformace na cluster
        """
        if self.clusters is None or self.line_points is None:
            raise ValueError("No clusters generated yet")

        # Konverze úhlů na radiány
        angles = np.deg2rad(rotation_angles)

        # Rotační matice pro každou osu
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])

        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])

        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])

        # Kombinovaná rotační matice
        R = Rz @ Ry @ Rx

        # Aplikace transformací
        self.transformed_points = self.clusters.copy()

        # Rotace
        self.transformed_points = (R @ self.transformed_points.T).T
        self.line_points = (R @ self.line_points.T).T

        # Škálování
        self.transformed_points *= scale
        self.line_points *= scale

        # Translace
        self.transformed_points += translation
        self.line_points += translation

    def get_line_points(self) -> np.ndarray:
        """
        Vrací aktuální body linie (po transformaci pokud byla aplikována)

        Returns:
        --------
        np.ndarray
            Body definující linii
        """
        return self.line_points if self.line_points is not None else self.waypoints

    def add_scatter_traces(self, fig: go.Figure, color: str, name_prefix: str = '',
                           show_waypoints: bool = True, show_centers: bool = True,
                           show_lines: bool = True):
        """
        Přidá stopy do Plotly grafu
        """
        if show_waypoints:
            fig.add_trace(go.Scatter3d(
                x=self.waypoints[:, 0],
                y=self.waypoints[:, 1],
                z=self.waypoints[:, 2],
                mode='markers',
                marker=dict(size=4, color='black'),
                name=f'{name_prefix} waypoints'
            ))

        if show_centers and self.line_points is not None:
            fig.add_trace(go.Scatter3d(
                x=self.line_points[:, 0],
                y=self.line_points[:, 1],
                z=self.line_points[:, 2],
                mode='markers',
                marker=dict(size=2, color='red'),
                name=f'{name_prefix} centers'
            ))

        if show_lines and self.line_points is not None:
            fig.add_trace(go.Scatter3d(
                x=self.line_points[:, 0],
                y=self.line_points[:, 1],
                z=self.line_points[:, 2],
                mode='lines',
                line=dict(color='black', width=1),
                name=f'{name_prefix} line'
            ))

        if self.transformed_points is not None:
            fig.add_trace(go.Scatter3d(
                x=self.transformed_points[:, 0],
                y=self.transformed_points[:, 1],
                z=self.transformed_points[:, 2],
                mode='markers',
                marker=dict(size=2, color=color),
                name=f'{name_prefix} cluster'
            ))


def generate_poisson_disk_nd(n_dimensions: int = 3,
                             dim_points: int = 100,
                             range_min: float = -20.0,
                             range_max: float = 20.0,
                             fill_percent: float = 0.3,
                             base_max_attempts: int = 200,
                             dtype: str = 'float') -> np.ndarray:
    """
    Generates n-dimensional Poisson disk sampling
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


def generate_line(num_points=10, length=1.0, center=(0, 0, 0)):
    """
    Generuje body na přímce o zadané délce.
    """
    t = np.linspace(-length / 2, length / 2, num_points)
    x = t
    y = np.zeros_like(t)
    z = np.zeros_like(t)

    cx, cy, cz = center
    x += cx
    y += cy
    z += cz

    return np.vstack((x, y, z)).T


def check_line_cluster_distance(line_points: np.ndarray,
                                radius: float,
                                existing_clusters: List[Tuple[np.ndarray, float]],
                                min_distance_factor: float = 0.75) -> bool:
    """
    Kontroluje minimální vzdálenost mezi liniovým clusterem a existujícími clustery
    """
    if not existing_clusters:
        return True

    for exist_center, exist_radius in existing_clusters:
        for point in line_points:
            distance = np.linalg.norm(point - exist_center)
            min_required_distance = (radius + exist_radius) * min_distance_factor

            if distance < min_required_distance:
                return False

    return True


def is_line_in_bounds(line_points: np.ndarray,
                      radius: float,
                      range_min: float,
                      range_max: float) -> bool:
    """
    Kontroluje, zda je celý liniový cluster v hranicích prostoru
    """
    return np.all((line_points - radius >= range_min) &
                  (line_points + radius <= range_max))


def save_datasets(points_dict: Dict,
                  n_dimensions: int,
                  output_dir: str = 'data/string') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Saves all datasets
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

    complete_df.to_csv(f'{output_dir}/string_cluster_dataset.csv', index=False)

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
        margin=dict(l=24, r=24, t=24, b=24),

        # Titulek grafu
        # title=f'Various shapes + Poisson-disk in BOX [-20,20], fill={fill_percent * 100:.0f}%',
        title=dict(
            text='Strings clusters + Poisson-disk noise in BOX [-20,20]',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font = dict(size=40)
        ),

        # Velikost výstupu
        width=1200,
        height=900,

        legend=dict(
            font=dict(size=20),
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
                titlefont=dict(size=24),
                tickfont=dict(size=18)
            ),
            yaxis=dict(
                range=[-20, 20],
                title='Y',
                titlefont=dict(size=24),
                tickfont=dict(size=18),
            ),
            zaxis=dict(
                range=[-20, 20],
                title='Z',
                titlefont=dict(size=24),
                tickfont=dict(size=18),
            )
        )
    )

def sphere_clusters_generator():
    """
    Main function for generating sphere clusters
    """
    fig = go.Figure()

    dd = int(800 / 4)

    rnd_position = [[2, 0.5, 1 * dd, 'blue'],
                    [4, 2, 2 * dd, 'red'],
                    [7, 3, 3 * dd, 'brown'],
                    [3, 2, 2 * dd, 'green'],
                    [12, 3, 3 * dd, 'magenta'],
                    [19, 1, 1 * dd, 'purple'],
                    [18, 4, 4 * dd, 'orange']]

    # Generate Poisson-disk points
    fill_percent = 0.9
    poisson_points = generate_poisson_disk_nd(
        n_dimensions=3, dim_points=666,
        range_min=-20, range_max=20,
        fill_percent=fill_percent, base_max_attempts=66)

    # Generate positions for cluster centers
    positions = generate_poisson_disk_nd(
        n_dimensions=1, dim_points=len(rnd_position),
        range_min=0, range_max=len(poisson_points),
        fill_percent=1, base_max_attempts=66, dtype='int')

    rotations = generate_poisson_disk_nd(
        n_dimensions=3, dim_points=len(rnd_position),
        range_min=0, range_max=180,
        fill_percent=1, base_max_attempts=66, dtype='int')

    # For data storage
    clusters_data = []
    position = 0
    used_indices = set()
    existing_clusters = []  # List for distance checking [(center, radius), ...]

    for (length, radius, density, color) in rnd_position:
        center_found = False
        current_idx = int(positions[position])
        attempts = 0
        max_attempts = 100

        while current_idx < len(poisson_points) and not center_found and attempts < max_attempts:
            if current_idx not in used_indices:
                center = poisson_points[current_idx]

                # Nejdřív vygenerujeme linii
                waypoint_line = generate_line(num_points=10, length=length, center=center)

                # Vytvoříme dočasný cluster pro kontrolu
                temp_cg = ClusterGenerator(waypoint_line, tension=0.165, cluster_density=density)
                temp_cg.generate_shape(tension=0.165, distribution='gauss', radius=radius)

                # Aplikujeme rotaci na dočasný cluster
                temp_cg.transform_clusters(rotations[position], [1, 1, 1], [0, 0, 0])

                # Získáme body transformované linie
                transformed_line = temp_cg.get_line_points()

                if (is_line_in_bounds(transformed_line, radius, -20, 20) and
                        check_line_cluster_distance(transformed_line, radius, existing_clusters)):

                    used_indices.add(current_idx)
                    center_found = True

                    # Přidáme cluster do seznamu pro kontrolu vzdáleností
                    existing_clusters.append((center, radius))

                    # Použijeme už vygenerovaný cluster
                    temp_cg.add_scatter_traces(
                        fig, color=color, name_prefix='Line',
                        show_waypoints=False, show_centers=False,
                        show_lines=False)

                    clusters_data.append((temp_cg.clusters, position))
                else:
                    if not is_line_in_bounds(transformed_line, radius, -20, 20):
                        print(f"Position {current_idx} invalid - line would exceed boundaries")
                    else:
                        print(f"Position {current_idx} invalid - too close to existing clusters")

            current_idx += 1
            attempts += 1

        if not center_found:
            print(f"Warning: Could not find suitable position for line with length {length} and radius {radius}")

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