import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pro 3D zobrazení
from lib.hdbscan import hdbscan


def generate_synthetic_data(num_points=50, dim=2, centers=[(0, 0), (10, 10)], spread=1.0):
    """
    Vygeneruje syntetická data v libovolném počtu dimenzí (dim).
    - centers: list center, každé centrum by mělo mít stejný počet dimenzí jako dim.
    - spread: rozptyl (std. dev) kolem center
    """
    data = []
    for c in centers:
        for _ in range(num_points):
            # Vygenerujeme 1 bod v 'dim' dimenzích
            point = tuple(
                random.gauss(c[d], spread)  # c[d] je souřadnice d-tého rozměru
                for d in range(dim)
            )
            data.append(point)
    return data


def plot_2d(data, labels, title="HDBSCAN (2D)"):
    """
    Jednoduchý scatter plot 2D dat podle cluster labelů s legendou.
    Vytvoří obrázek ~1920x1080 px (figure + dpi).
    """
    # Nastavíme velikost a DPI
    plt.figure(figsize=(19.2, 10.8), dpi=100)

    # Seskupíme body podle labelu
    label_to_points = {}
    for i, lab in enumerate(labels):
        label_to_points.setdefault(lab, []).append(data[i])

    # Vykreslíme každý cluster samostatným voláním scatter
    for lab, points in label_to_points.items():
        # Připravíme si souřadnice x, y
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Šum = -1 => vykreslit černě a jako "Noise", jinak náhodná barva a "Cluster X"
        if lab == -1:
            color = (0, 0, 0)
            label_str = "Noise"
        else:
            color = (random.random(), random.random(), random.random())
            label_str = f"Cluster {lab}"

        plt.scatter(xs, ys, c=[color], label=label_str, s=30)

    plt.title(title)
    plt.legend(loc='best')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def plot_3d(data, labels, title="HDBSCAN (3D)"):
    """
    Jednoduchý scatter plot 3D dat podle cluster labelů s legendou.
    Vytvoří obrázek ~1920x1080 px.
    """
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Seskupíme body podle labelu
    label_to_points = {}
    for i, lab in enumerate(labels):
        label_to_points.setdefault(lab, []).append(data[i])

    for lab, points in label_to_points.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        if lab == -1:
            color = (0, 0, 0)
            label_str = "Noise"
        else:
            color = (random.random(), random.random(), random.random())
            label_str = f"Cluster {lab}"

        ax.scatter(xs, ys, zs, c=[color], label=label_str, s=30)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # U 3D scatteru je legend() také dostupný, ale občas ne vždy hezky zobrazený:
    ax.legend(loc='best')
    plt.show()


def main():
    # 1) Příklad 2D
    dim2 = 2
    data_2d = generate_synthetic_data(
        num_points=50,
        dim=dim2,
        centers=[(0, 0), (10, 10)],
        spread=1.5
    )
    labels_2d = hdbscan(data_2d, min_points=5, min_cluster_size=5)
    plot_2d(data_2d, labels_2d, title="HDBSCAN výsledek (2D)")

    # 2) Příklad 3D
    dim3 = 3
    data_3d = generate_synthetic_data(
        num_points=50,
        dim=dim3,
        centers=[(0, 0, 0), (10, 10, 10)],
        spread=1.5
    )
    labels_3d = hdbscan(data_3d, min_points=5, min_cluster_size=5)
    plot_3d(data_3d, labels_3d, title="HDBSCAN výsledek (3D)")


if __name__ == "__main__":
    main()
