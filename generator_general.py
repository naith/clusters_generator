import numpy as np
import plotly.graph_objects as go

from lib.cluster_generator import ClusterGenerator


# --------------------------------------------------
# NOVÉ FUNKCE: generování waypointů
# --------------------------------------------------
def generate_modulated_ring(num_points=10, radius=1.0, z_amplitude=0.2,
                            freq=3, center=(0, 0, 0)):
    """
    Vytvoří prstenec (kruh v rovině XY) s modulací sinusovkou v ose Z,
    posunutý o vektor center = (cx, cy, cz).
    """
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = z_amplitude * np.sin(freq * t)

    # Posun do (cx, cy, cz)
    cx, cy, cz = center
    x += cx
    y += cy
    z += cz

    return np.vstack((x, y, z)).T


def generate_semi_circle(num_points=10, radius=1.0, z_offset=0.0,
                         center=(0, 0, 0)):
    """
    Vygeneruje půlkruh v rovině XY (z=0),
    pak posune nahoru o z_offset (původní volba),
    a nakonec posune o vektor center = (cx, cy, cz).
    """
    t = np.linspace(0, np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros_like(x) + z_offset

    # Posun do (cx, cy, cz)
    cx, cy, cz = center
    x += cx
    y += cy
    z += cz

    return np.vstack((x, y, z)).T


def generate_wild_shape(num_points=10, center=(0, 0, 0)):
    """
    Trošku 'divočejší' 3D spirála,
    zkrácená na interval t ∈ [0, 2π].
    Navíc posun o vektor center = (cx, cy, cz).
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


def generate_crazy_spiral_waypoints(num_points=50,
                                    dimension=3,
                                    spiral_turns=3,
                                    noise_amplitude=0.2,
                                    center=None):
    if dimension < 2:
        raise ValueError("Dimension must be at least 2")

    if center is None:
        center = np.zeros(dimension)
    elif len(center) != dimension:
        raise ValueError("Center must match the dimension")

    # Parametr t
    t = np.linspace(0, 2 * np.pi * spiral_turns, num_points)

    # Inicializace
    points = np.zeros((num_points, dimension))

    # Základní vzorce pro první 3 dimenze
    points[:, 0] = t * np.cos(t)  # x = t * cos(t)
    points[:, 1] = t * np.sin(t)  # y = t * sin(t)
    if dimension > 2:
        points[:, 2] = 0.5 * t  # z = 0.5 * t (pro třetí dimenzi)

    # Dimenze > 3: ostatní dimenze jsou náhodné sinusovky
    for d in range(3, dimension):
        points[:, d] = np.sin((d + 1) * t / spiral_turns)

    # Přidání šumu
    noise = noise_amplitude * np.random.randn(num_points, dimension)
    points += noise

    # Posun do středu
    points += np.array(center)

    return points


# --------------------------------------------------
# 3D POISSON-DISK SAMPLING
# (dists = None -> fix na 1. bod)
# --------------------------------------------------
def generate_poisson_disk_3d(fill_percent=0.3,
                             max_points=2000, max_attempts=200):
    """
    V této verzi natvrdo použijeme bounding box = [-20,20]^3.
    fill_percent => určuje hustotu (čím větší, tím hustší).
    """
    bbox_min = np.array([-20, -20, -20], dtype=float)
    bbox_max = np.array([20, 20, 20], dtype=float)

    diag = np.linalg.norm(bbox_max - bbox_min)
    # Čím větší fill_percent, tím menší r => hustší body
    r = diag * (0.5 - 0.49 * fill_percent)

    points = []
    attempts_count = 0

    while len(points) < max_points and attempts_count < max_attempts:
        candidate = np.random.uniform(bbox_min, bbox_max)
        dists = None  # fix pro 1. bod

        if len(points) == 0:
            points.append(candidate)
        else:
            dists = np.linalg.norm(np.array(points) - candidate, axis=1)
            if np.all(dists >= r):
                points.append(candidate)

        attempts_count += 1
        # Pokud jsme bod úspěšně přidali, reset pokusů
        if len(points) > 0:
            # Pro 1. bod (dists=None) => True, nebo pro dalším bod, kde platí dists>=r
            if (dists is None) or np.all(dists >= r):
                attempts_count = 0

    return np.array(points)


# --------------------------------------------------
# DEMO: Tři různé křivky v JEDNOM GRAFU + Poisson-disk
# --------------------------------------------------
def demo_three_curves_in_one_figure():
    # 2) Catmull-Rom + Clustery
    N = 1000
    tension = 0.005
    cluster_size = 20
    cluster_size2 = 70
    distribution = 'gauss'
    diameter_factor = 5.0
    diameter_factor2 = 30.0
    M = 100


    fig = go.Figure()

    # Ring cluster
    waypoints_ring = generate_modulated_ring(num_points=12, radius=1.5,
                                             z_amplitude=1.2, freq=3,
                                             center=(-7, -10, 10))

    cg_ring = ClusterGenerator(waypoints_ring, tension=0.165, cluster_size=20,
                               diameter_factor=5.0)
    cg_ring.transform_nd_axes([0, 0, 0], [1, 1, 1], [0, 0, 0])
    cg_ring.generate_shape(tension=0.165, distribution='gauss',
                           total_samples=100)
    cg_ring.add_scatter_traces(fig, color='blue', name_prefix='Ring')

    # Semi circle cluster
    waypoints_semi = generate_semi_circle(num_points=12, radius=2.0,
                                          z_offset=2.0, center=(-5, -2, -10))
    cg_semi = ClusterGenerator(waypoints_semi, tension=0.165, cluster_size=20,
                               diameter_factor=5.0)
    cg_semi.transform_nd_axes([0, 0, 0], [1, 1, 1], [0, 0, 0])
    cg_semi.generate_shape(tension=0.165, distribution='gauss',
                           total_samples=100)
    cg_semi.add_scatter_traces(fig, color='red', name_prefix='Semi')


    # Wild shape cluster
    waypoints_wild = generate_wild_shape(num_points=12, center=(5, -2, 10))
    cg_wild = ClusterGenerator(waypoints_wild, tension=0.165, cluster_size=20,
                               diameter_factor=5.0)
    cg_wild.transform_nd_axes([0, 0, 0], [1, 1, 1], [0, 0, 0])
    cg_wild.generate_shape(tension=0.165, distribution='gauss',
                           total_samples=100)
    cg_wild.add_scatter_traces(fig, color='green', name_prefix='Wild')


    # Wild shape cluster2
    waypoints_wild2 = generate_wild_shape(num_points=12, center=(-10, 12, -10))
    cg_wild2 = ClusterGenerator(waypoints_wild2, tension=0.165,
                                cluster_size=20, diameter_factor=5.0)
    cg_wild2.transform_nd_axes([45, 45, 45], [1, 1, 1], [0, 0, 0])
    cg_wild2.generate_shape(tension=0.165, distribution='gauss',
                            total_samples=1000)
    cg_wild2.add_scatter_traces(fig, color='purple', name_prefix='Wild2')


    # Wild shape cluster3
    waypoints_wild3 = generate_crazy_spiral_waypoints(24, 3, 3, 0.2,
                                                      [10, 3, -4])
    cg_wild3 = ClusterGenerator(waypoints_wild3, tension=0.165,
                                cluster_size=170, diameter_factor=1.0)
    cg_wild3.transform_nd_axes([0, 30, 0], [0.3, 0.3, 1], [0, 0, 0])
    cg_wild3.generate_shape(tension=0.165, distribution='gauss',
                            total_samples=100)
    cg_wild3.add_scatter_traces(fig, color='orange', name_prefix='Wild3')


    # Wild shape cluster4
    waypoints_wild4 = generate_crazy_spiral_waypoints(24, 3, 1, 1, [10, -3, 4])
    cg_wild4 = ClusterGenerator(waypoints_wild4, tension=0.165,
                                cluster_size=70, diameter_factor=1.0)
    cg_wild4.transform_nd_axes([0, 30, 0], [0.3, 0.3, 1], [0, 0, 0])
    cg_wild4.generate_shape(tension=0.165, distribution='gauss',
                            total_samples=100)
    cg_wild4.add_scatter_traces(fig, color='cyan', name_prefix='Wild4')

    # 3) Generujeme Poisson-disk body pro BOX [-20,20]^3
    fill_percent = 0.8
    poisson_points = generate_poisson_disk_3d(
        fill_percent=fill_percent,
        max_points=4000,
        max_attempts=500
    )

    # (d) Poisson-disk (gray)
    fig.add_trace(go.Scatter3d(
        x=poisson_points[:, 0],
        y=poisson_points[:, 1],
        z=poisson_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='gray'),
        name='Poisson-disk (gray)'
    ))

    fig.update_layout(
        title=f'Tři různé křivky + Poisson-disk  v BOX [-20,20], fill={fill_percent * 100:.0f}%',
        width=900,
        height=800
    )

    fig.update_layout(
        scene=dict(
            # aspectmode='cube' nebo 'manual' dle chuti
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[-20, 20], title='X'),
            yaxis=dict(range=[-20, 20], title='Y'),
            zaxis=dict(range=[-20, 20], title='Z')
        )
    )
    fig.show()


# --------------------------------------------------
# Spuštění
# --------------------------------------------------
if __name__ == "__main__":
    # Původní spirála:
    # demo_string_clusters_constant_spread()

    # Nové tři křivky + Poisson-disk [-20,20]^3
    demo_three_curves_in_one_figure()
