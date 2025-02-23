import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.interpolate import CubicSpline


def create_directories(path: str) -> None:
    """Creates directory structure"""
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_circle(num_points: int = 30,
                    radius: float = 1.0,
                    center: Tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    """
    Generuje body na kružnici o zadaném poloměru.

    Parameters:
    -----------
    num_points : int
        Počet bodů na kružnici
    radius : float
        Poloměr kružnice
    center : tuple
        Střed kružnice (cx, cy, cz)

    Returns:
    --------
    np.ndarray
        Body kružnice tvaru (num_points, 3)
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros_like(t)

    cx, cy, cz = center
    x += cx
    y += cy
    z += cz

    return np.vstack((x, y, z)).T


class RingClusterGenerator:
    """
    Třída pro generování a správu prstencových clusterů
    """

    def __init__(self, ring_points: np.ndarray, tension: float = 0.5, cluster_density: int = 100):
        """
        Inicializace generátoru clusterů

        Parameters:
        -----------
        ring_points : np.ndarray
            Body definující tvar prstence
        tension : float
            Napětí křivky (0-1)
        cluster_density : int
            Hustota bodů v clusteru
        """
        self.ring_points = ring_points
        self.tension = tension
        self.cluster_density = cluster_density
        self.clusters = None
        self.transformed_points = None
        self.ring_points_transformed = None

    def generate_shape(self,
                       tension: float,
                       distribution: str = 'uniform',
                       radius: float = 1.0,
                       close_curve: bool = True):
        """
        Generuje body clusteru podél prstence

        Parameters:
        -----------
        tension : float
            Napětí křivky
        distribution : str
            Typ distribuce bodů ('uniform' nebo 'gaussian')
        radius : float
            Poloměr clusteru kolem prstence
        close_curve : bool
            Zda uzavřít křivku (True pro prstenec)
        """
        # Přidáme první bod na konec pro uzavření křivky
        if close_curve:
            points = np.vstack([self.ring_points, self.ring_points[0]])
        else:
            points = self.ring_points

        # Interpolace bodů podél křivky
        t = np.linspace(0, 1, self.cluster_density)
        cs = CubicSpline(np.linspace(0, 1, len(points)), points, axis=0)
        self.ring_points_transformed = cs(t)

        # Generování bodů clusteru
        clusters = []
        for point in self.ring_points_transformed:
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
        if self.clusters is None or self.ring_points_transformed is None:
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
        self.ring_points_transformed = (R @ self.ring_points_transformed.T).T

        # Škálování
        self.transformed_points *= scale
        self.ring_points_transformed *= scale

        # Translace
        self.transformed_points += translation
        self.ring_points_transformed += translation

    def get_ring_points(self) -> np.ndarray:
        """
        Vrací aktuální body prstence (po transformaci pokud byla aplikována)
        """
        return self.ring_points_transformed if self.ring_points_transformed is not None else self.ring_points

    def add_scatter_traces(self,
                           fig: go.Figure,
                           color: str,
                           name_prefix: str = '',
                           show_ring: bool = True,
                           show_points: bool = True):
        """
        Přidá stopy do Plotly grafu
        """
        # if show_ring and self.ring_points_transformed is not None:
        #     fig.add_trace(go.Scatter3d(
        #         x=self.ring_points_transformed[:, 0],
        #         y=self.ring_points_transformed[:, 1],
        #         z=self.ring_points_transformed[:, 2],
        #         mode='lines',
        #         line=dict(color='black', width=1),
        #         name=f'{name_prefix} ring'
        #     ))

        if show_points and self.transformed_points is not None:
            fig.add_trace(go.Scatter3d(
                x=self.transformed_points[:, 0],
                y=self.transformed_points[:, 1],
                z=self.transformed_points[:, 2],
                mode='markers',
                marker=dict(size=2, color=color),
                name=f'{name_prefix} cluster'
            ))


def check_ring_collision(ring_points1: np.ndarray,
                         ring_points2: np.ndarray,
                         radius1: float,
                         radius2: float,
                         min_distance_factor: float = 0.75) -> bool:
    """
    Kontroluje kolize mezi dvěma prstenci

    Parameters:
    -----------
    ring_points1, ring_points2 : np.ndarray
        Body definující prstence
    radius1, radius2 : float
        Poloměry clusterů kolem prstenců
    min_distance_factor : float
        Minimální požadovaná vzdálenost jako násobek součtu poloměrů

    Returns:
    --------
    bool
        True pokud nedochází ke kolizi, False jinak
    """
    min_required_distance = (radius1 + radius2) * min_distance_factor

    # Pro optimalizaci vybereme méně bodů pro rychlou kontrolu
    n_check_points = 10

    # Výběr bodů pro první prstenec
    step1 = max(1, len(ring_points1) // n_check_points)
    selected_points1 = ring_points1[::step1]

    # Výběr bodů pro druhý prstenec
    step2 = max(1, len(ring_points2) // n_check_points)
    selected_points2 = ring_points2[::step2]

    # Rychlá kontrola na vybraných bodech
    for p1 in selected_points1:
        distances = np.linalg.norm(selected_points2 - p1, axis=1)
        if np.any(distances < min_required_distance):
            return False

    # Detailnější kontrola na všech bodech
    step_detailed = max(1, len(ring_points1) // 20)  # Použijeme víc bodů pro detailní kontrolu
    for p1 in ring_points1[::step_detailed]:
        distances = np.linalg.norm(ring_points2 - p1, axis=1)
        if np.any(distances < min_required_distance):
            return False

    return True


def is_ring_in_bounds(ring_points: np.ndarray,
                      radius: float,
                      range_min: float,
                      range_max: float) -> bool:
    """
    Kontroluje, zda je celý prstencový cluster v hranicích prostoru
    """
    return np.all((ring_points - radius >= range_min) &
                  (ring_points + radius <= range_max))


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


def save_datasets(points_dict: Dict,
                  n_dimensions: int,
                  output_dir: str = 'data/rings') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    complete_df.to_csv(f'{output_dir}/ring_cluster_dataset.csv', index=False)

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
            text='Rings clusters + Poisson-disk noise in BOX [-20,20]',
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

def ring_clusters_generator():
    """
    Main function for generating ring clusters
    """
    fig = go.Figure()

    dd = int(800 / 2)

    # [radius_kruhu, radius_clusteru, hustota, barva]
    rnd_position = [
        [5, 2, 1 * dd, 'blue'],  # malý prstenec
        [7, 3, 2 * dd, 'red'],  # střední prstenec
        [9, 4, 3 * dd, 'green'],  # velký prstenec
        [4, 1, 2 * dd, 'purple'],  # tenký prstenec
        [6, 2, 2 * dd, 'orange'],  # střední prstenec
        [8, 3, 3 * dd, 'magenta'],  # široký prstenec
        [5, 1, 2 * dd, 'brown']  # malý tenký prstenec
    ]

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
    existing_rings = []  # Seznam vygenerovaných prstenců pro kontrolu kolizí

    for (ring_radius, cluster_radius, density, color) in rnd_position:
        center_found = False
        current_idx = int(positions[position])
        attempts = 0
        max_attempts = 100

        while current_idx < len(poisson_points) and not center_found and attempts < max_attempts:
            if current_idx not in used_indices:
                center = poisson_points[current_idx]

                # Nejdřív vygenerujeme prstenec
                ring_points = generate_circle(num_points=30, radius=ring_radius, center=center)

                # Vytvoříme dočasný cluster pro kontrolu
                temp_cg = RingClusterGenerator(ring_points, tension=0.165, cluster_density=density)
                temp_cg.generate_shape(tension=0.165, distribution='gauss', radius=cluster_radius)

                # Aplikujeme rotaci na dočasný cluster
                temp_cg.transform_clusters(rotations[position], [1, 1, 1], [0, 0, 0])

                # Získáme body transformovaného prstence
                transformed_ring = temp_cg.get_ring_points()

                # Kontrola hranic a kolizí
                valid_position = is_ring_in_bounds(transformed_ring, cluster_radius, -20, 20)
                if valid_position and existing_rings:
                    for existing_ring, exist_radius in existing_rings:
                        if not check_ring_collision(transformed_ring, existing_ring,
                                                    cluster_radius, exist_radius):
                            valid_position = False
                            break

                if valid_position:
                    used_indices.add(current_idx)
                    center_found = True

                    # Přidáme prstenec do seznamu pro kontrolu kolizí
                    existing_rings.append((transformed_ring, cluster_radius))

                    # Použijeme už vygenerovaný cluster
                    temp_cg.add_scatter_traces(
                        fig, color=color, name_prefix=f'Ring_{position}',
                        show_ring=True, show_points=True)

                    clusters_data.append((temp_cg.clusters, position))
                else:
                    if not is_ring_in_bounds(transformed_ring, cluster_radius, -20, 20):
                        print(f"Position {current_idx} invalid - ring would exceed boundaries")
                    else:
                        print(f"Position {current_idx} invalid - collision with existing rings")

            current_idx += 1
            attempts += 1

        if not center_found:
            print(f"Warning: Could not find suitable position for ring with radius {ring_radius}")

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
    ring_clusters_generator()