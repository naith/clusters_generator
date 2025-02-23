import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.interpolate import CubicSpline

show_lines = False

def create_directories(path: str) -> None:
    """Creates directory structure"""
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_butterfly_trajectory(num_points=30, center=(0, 0, 0)):
    """
    Generates a butterfly-like flight trajectory
    Combines sinusoidal movement with spiral elements and random variations
    """
    t = np.linspace(0, 4 * np.pi, num_points)

    # Base spiraling movement
    x = t * np.cos(t / 2) * 0.3

    # Vertical oscillation with varying amplitude
    y = np.sin(2 * t) * (1 + 0.3 * np.cos(t / 2))

    # Forward movement with height variation
    z = t / 2 + np.sin(3 * t) * 0.5

    # Add some natural variation
    x += np.sin(5 * t) * 0.2
    y += np.sin(4 * t) * 0.15
    z += np.sin(6 * t) * 0.1

    cx, cy, cz = center
    x += cx
    y += cy
    z += cz

    return np.vstack((x, y, z)).T


def generate_modulated_ring(num_points=10, radius=1.0, z_amplitude=0.2,
                            freq=3, center=(0, 0, 0)):
    """
    Generates a modulated ring (wavy in Z-axis)
    """
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = z_amplitude * np.sin(freq * t)

    cx, cy, cz = center
    x += cx
    y += cy
    z += cz

    return np.vstack((x, y, z)).T


def generate_semi_circle(num_points=10, radius=1.0, z_offset=0.0,
                         center=(0, 0, 0)):
    """
    Generates a semi-circle
    """
    t = np.linspace(0, np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros_like(x) + z_offset

    cx, cy, cz = center
    x += cx
    y += cy
    z += cz

    return np.vstack((x, y, z)).T


def generate_custom_curve(num_points=10, center=(0, 0, 0)):
    """
    Generates a custom curve shape using trigonometric functions
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = t
    y = np.sin(3 * t)
    z = t + 0.3 * np.sin(5 * t)

    cx, cy, cz = center
    x += cx
    y += cy
    z += cz

    return np.vstack((x, y, z)).T


def generate_spiral_waypoints(num_points=50, dimension=3, spiral_turns=3,
                              noise_amplitude=0.2, center=None):
    """
    Generates a spiral with optional noise
    """
    if dimension < 2:
        raise ValueError("Dimension must be at least 2")

    if center is None:
        center = np.zeros(dimension)
    elif len(center) != dimension:
        raise ValueError("Center must match the dimension")

    t = np.linspace(0, 2 * np.pi * spiral_turns, num_points)
    points = np.zeros((num_points, dimension))

    points[:, 0] = t * np.cos(t)
    points[:, 1] = t * np.sin(t)
    if dimension > 2:
        points[:, 2] = 0.5 * t

    for d in range(3, dimension):
        points[:, d] = np.sin((d + 1) * t / spiral_turns)

    noise = noise_amplitude * np.random.randn(num_points, dimension)
    points += noise
    points += np.array(center)

    return points


def generate_sphere_waypoints(center=(0, 0, 0)):
    """
    Generates a point for spherical cluster
    """
    return np.array([center])


class ClusterGenerator:
    """
    Class for generating and managing clusters
    """

    def __init__(self, waypoints: np.ndarray, tension: float = 0.5, cluster_density: int = 100):
        self.waypoints = waypoints
        self.tension = tension
        self.cluster_density = cluster_density
        self.clusters = None
        self.transformed_points = None
        self.shape_points = None

    def generate_shape(self, tension: float, distribution: str = 'uniform', radius: float = 1.0):
        """
        Generates cluster points along the shape or around a central point for spheres
        """
        # Special handling for sphere (single point)
        if len(self.waypoints) == 1:
            self.shape_points = self.waypoints
            clusters = []
            center = self.waypoints[0]

            for _ in range(self.cluster_density):
                if distribution == 'uniform':
                    phi = np.random.uniform(0, 2 * np.pi)
                    cos_theta = np.random.uniform(-1, 1)
                    theta = np.arccos(cos_theta)
                    r = radius * np.random.uniform(0, 1) ** (1 / 3)

                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)

                    cluster_point = center + np.array([x, y, z])

                else:  # gaussian
                    noise = np.random.normal(0, radius / 3, 3)
                    cluster_point = center + noise

                clusters.append(cluster_point)

        # Normal handling for curves and other shapes
        else:
            t = np.linspace(0, 1, self.cluster_density)
            cs = CubicSpline(np.linspace(0, 1, len(self.waypoints)), self.waypoints, axis=0)
            self.shape_points = cs(t)

            clusters = []
            for point in self.shape_points:
                if distribution == 'uniform':
                    phi = np.random.uniform(0, 2 * np.pi)
                    cos_theta = np.random.uniform(-1, 1)
                    theta = np.arccos(cos_theta)
                    r = radius * np.random.uniform(0, 1) ** (1 / 3)

                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)

                    cluster_point = point + np.array([x, y, z])

                else:  # gaussian
                    noise = np.random.normal(0, radius / 3, 3)
                    cluster_point = point + noise

                clusters.append(cluster_point)

        self.clusters = np.array(clusters)
        self.transformed_points = self.clusters.copy()

    def transform_clusters(self, rotation_angles: np.ndarray, scale: List[float], translation: List[float]):
        """
        Applies transformations to the cluster
        """
        if self.clusters is None or self.shape_points is None:
            raise ValueError("No clusters generated yet")

        angles = np.deg2rad(rotation_angles)

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])

        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])

        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx

        self.transformed_points = self.clusters.copy()

        self.transformed_points = (R @ self.transformed_points.T).T
        self.shape_points = (R @ self.shape_points.T).T

        self.transformed_points *= scale
        self.shape_points *= scale

        self.transformed_points += translation
        self.shape_points += translation

    def get_shape_points(self) -> np.ndarray:
        """
        Returns current shape points (after transformation if applied)
        """
        return self.shape_points if self.shape_points is not None else self.waypoints

    def add_scatter_traces(self, fig: go.Figure, color: str, name_prefix: str = '',
                           show_shape: bool = True, show_points: bool = True):
        """
        Adds traces to Plotly figure
        """
        if show_shape and self.shape_points is not None and show_lines is True:
            fig.add_trace(go.Scatter3d(
                x=self.shape_points[:, 0],
                y=self.shape_points[:, 1],
                z=self.shape_points[:, 2],
                mode='lines',
                line=dict(color='black', width=1),
                name=f'{name_prefix} shape'
            ))

        if show_points and self.transformed_points is not None:
            fig.add_trace(go.Scatter3d(
                x=self.transformed_points[:, 0],
                y=self.transformed_points[:, 1],
                z=self.transformed_points[:, 2],
                mode='markers',
                marker=dict(size=2, color=color),
                name=f'{name_prefix} cluster'
            ))


def check_shape_collision(shape_points1: np.ndarray,
                          shape_points2: np.ndarray,
                          radius1: float,
                          radius2: float,
                          min_distance_factor: float = 0.75) -> bool:
    """
    Checks collisions between two shapes
    """
    min_required_distance = (radius1 + radius2) * min_distance_factor

    n_check_points = 10

    step1 = max(1, len(shape_points1) // n_check_points)
    selected_points1 = shape_points1[::step1]

    step2 = max(1, len(shape_points2) // n_check_points)
    selected_points2 = shape_points2[::step2]

    for p1 in selected_points1:
        distances = np.linalg.norm(selected_points2 - p1, axis=1)
        if np.any(distances < min_required_distance):
            return False

    step_detailed = max(1, len(shape_points1) // 20)
    for p1 in shape_points1[::step_detailed]:
        distances = np.linalg.norm(shape_points2 - p1, axis=1)
        if np.any(distances < min_required_distance):
            return False

    return True


def is_shape_in_bounds(shape_points: np.ndarray,
                       radius: float,
                       range_min: float,
                       range_max: float) -> bool:
    """
    Checks if the entire shape is within space bounds
    """
    return np.all((shape_points - radius >= range_min) &
                  (shape_points + radius <= range_max))


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
                  output_dir: str = 'data/shapes') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    complete_df.to_csv(f'{output_dir}/shapes_dataset.csv', index=False)

    return background_df, clusters_df, complete_df


def visualize_config(fig, poisson_points, fill_percent):
    fig.add_trace(go.Scatter3d(
        x=poisson_points[:, 0],
        y=poisson_points[:, 1],
        z=poisson_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='gray'),
        name='Poisson-disk (gray)'
    ))

    fig.update_layout(
        # Zmenšíme okraje kolem celé plochy
        margin=dict(l=20, r=20, t=20, b=20),

        # Titulek grafu
        # title=f'Various shapes + Poisson-disk in BOX [-20,20], fill={fill_percent * 100:.0f}%',
        title=dict(
            text='Various shapes + Poisson-disk in BOX [-20,20]',
            x=0.5,  # zarovnat na střed
            y=0.95,  # snížit či zvýšit (1.0 = úplně nahoře, menší => níž)
            xanchor='center',
            yanchor='top',
            font = dict(size=20)
        ),

        # Velikost výstupu
        width=1200,
        height=900,
        legend=dict(
            font=dict(size=16),  # zvolte si velikost písma
            x=1.0,  # vodorovná pozice (0 = vlevo, 1 = vpravo)
            y=0.0,  # svislá pozice (0 = dole, 1 = nahoře)
            xanchor='right',  # ukotvení legendy ve vodorovném směru (k pravému okraji)
            yanchor='bottom',  # ukotvení legendy ve svislém směru (k dolnímu okraji)
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
                titlefont=dict(size=16),  # <--- menší standoff = bližší label
                tickfont=dict(size=14)  # (volitelně menší font)
            ),
            yaxis=dict(
                range=[-20, 20],
                title='Y',
                titlefont=dict(size=16),  # <--- menší standoff = bližší label
                tickfont=dict(size=14),
            ),
            zaxis=dict(
                range=[-20, 20],
                title='Z',
                titlefont=dict(size=16),  # <--- menší standoff = bližší label
                tickfont=dict(size=14),
            )
        )
    )


def multishape_clusters_generator():
    """
    Main function for generating various shape clusters
    """
    fig = go.Figure()



    dd = int(800 / 2)

    # [shape_type, shape_params, cluster_radius, density, color]
    rnd_position = [
        # Butterfly trajectory
        ['butterfly', {}, 1.0, 4 * dd, 'lightgreen'],

        # Modulated rings
        ['modulated_ring', {'radius': 5.5, 'z_amplitude': 5.5, 'freq': 4}, 3.0, 3 * dd, 'blue'],
        ['modulated_ring', {'radius': 4.0, 'z_amplitude': 3.0, 'freq': 3}, 2.0, 2 * dd, 'red'],

        # Semi-circles
        ['semi_circle', {'radius': 4.0, 'z_offset': 2.0}, 1.2, 1 * dd, 'green'],
        ['semi_circle', {'radius': 3.0, 'z_offset': 1.5}, 1.0, 1 * dd, 'brown'],

        # Custom curves
        ['custom_curve', {}, 1.0, 1 * dd, 'purple'],
        ['custom_curve', {}, 1.2, 1 * dd, 'orange'],

        # Spirals
        ['spiral', {'spiral_turns': 3, 'noise_amplitude': 0.2}, 2.75, 3 * dd, 'magenta'],
        ['spiral', {'spiral_turns': 2, 'noise_amplitude': 0.3}, 2.0, 2 * dd, 'cyan'],

        # Spheres
        ['sphere', {}, 7.0, 7 * dd, 'black'],
        ['sphere', {}, 4.0, 4 * dd, 'yellow'],
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
    existing_shapes = []  # List for distance checking [(shape_points, radius), ...]

    for (shape_type, shape_params, cluster_radius, density, color) in rnd_position:
        center_found = False
        current_idx = int(positions[position])
        attempts = 0
        max_attempts = 100

        while current_idx < len(poisson_points) and not center_found and attempts < max_attempts:
            if current_idx not in used_indices:
                center = poisson_points[current_idx]

                # Generate points based on shape type
                if shape_type == 'modulated_ring':
                    waypoints = generate_modulated_ring(
                        num_points=30,
                        radius=shape_params['radius'],
                        z_amplitude=shape_params['z_amplitude'],
                        freq=shape_params['freq'],
                        center=center
                    )
                elif shape_type == 'semi_circle':
                    waypoints = generate_semi_circle(
                        num_points=30,
                        radius=shape_params['radius'],
                        z_offset=shape_params['z_offset'],
                        center=center
                    )
                elif shape_type == 'custom_curve':
                    waypoints = generate_custom_curve(
                        num_points=30,
                        center=center
                    )
                elif shape_type == 'spiral':
                    waypoints = generate_spiral_waypoints(
                        num_points=30,
                        spiral_turns=shape_params['spiral_turns'],
                        noise_amplitude=shape_params['noise_amplitude'],
                        center=center
                    )
                elif shape_type == 'butterfly':
                    waypoints = generate_butterfly_trajectory(
                        num_points=30,
                        center=center
                    )
                else:  # sphere
                    waypoints = generate_sphere_waypoints(center)

                # Create temporary cluster for validation
                temp_cg = ClusterGenerator(waypoints, tension=0.165, cluster_density=density)
                temp_cg.generate_shape(tension=0.165, distribution='gauss', radius=cluster_radius)
                scace = [1, 1, 1]
                if shape_type == 'butterfly':
                    scace = [2, 2, 2]
                # Apply rotation to temporary cluster
                temp_cg.transform_clusters(rotations[position], scace, [0, 0, 0])

                # Get transformed shape points
                transformed_shape = temp_cg.get_shape_points()

                # Check bounds and collisions
                valid_position = is_shape_in_bounds(transformed_shape, cluster_radius, -20, 20)
                if valid_position and existing_shapes:
                    for existing_shape, exist_radius in existing_shapes:
                        if not check_shape_collision(transformed_shape, existing_shape,
                                                     cluster_radius, exist_radius):
                            valid_position = False
                            break

                if valid_position:
                    used_indices.add(current_idx)
                    center_found = True

                    # Add shape to list for collision checking
                    existing_shapes.append((transformed_shape, cluster_radius))

                    # Use the generated cluster
                    temp_cg.add_scatter_traces(
                        fig, color=color, name_prefix=f'{shape_type}_{position}',
                        show_shape=True, show_points=True)

                    clusters_data.append((temp_cg.clusters, position))
                else:
                    if not is_shape_in_bounds(transformed_shape, cluster_radius, -20, 20):
                        print(f"Position {current_idx} invalid - shape would exceed boundaries")
                    else:
                        print(f"Position {current_idx} invalid - collision with existing shapes")

            current_idx += 1
            attempts += 1

        if not center_found:
            print(f"Warning: Could not find suitable position for {shape_type}")

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
    multishape_clusters_generator()
