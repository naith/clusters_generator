import numpy as np
import pandas as pd
import plotly.express as px
from typing import List, Tuple
from lib.dbscan import DBSCAN

def generate_test_data(n_points: int = 100, n_dims: int = 2) -> List[List[float]]:
    """
    Generuje testovací data pro DBSCAN.
    Vytvoří dva shluky bodů a několik náhodných bodů jako šum.

    Args:
        n_points: Počet bodů k vygenerování
        n_dims: Počet dimenzí

    Returns:
        List bodů pro testování
    """
    # První cluster
    cluster1 = np.random.normal(0, 1, (n_points // 3, n_dims))

    # Druhý cluster
    cluster2 = np.random.normal(5, 1, (n_points // 3, n_dims))

    # Náhodný šum
    noise = np.random.uniform(-2, 7, (n_points // 3, n_dims))

    # Spojení všech bodů
    X = np.vstack([cluster1, cluster2, noise])

    return X.tolist()


def visualize_clusters(X: List[List[float]], labels: List[int]) -> None:
    """
    Vizualizuje výsledky clusteringu pomocí plotly.
    Funguje pro 2D a 3D data.

    Args:
        X: Dataset
        labels: Výsledky clusteringu
    """
    X_np = np.array(X)
    if X_np.shape[1] == 2:
        df = pd.DataFrame({
            'x': X_np[:, 0],
            'y': X_np[:, 1],
            'cluster': [f'Cluster {l}' if l >= 0 else 'Noise' for l in labels]
        })

        fig = px.scatter(df, x='x', y='y', color='cluster',
                         title='DBSCAN Clustering Results (2D)')
        fig.show()

    elif X_np.shape[1] == 3:
        df = pd.DataFrame({
            'x': X_np[:, 0],
            'y': X_np[:, 1],
            'z': X_np[:, 2],
            'cluster': [f'Cluster {l}' if l >= 0 else 'Noise' for l in labels]
        })

        fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster',
                            title='DBSCAN Clustering Results (3D)')
        fig.show()

    else:
        print(f"Vizualizace není podporována pro {X_np.shape[1]} dimenzí")


def save_results(X: List[List[float]], labels: List[int], filename: str = 'dbscan_results.csv') -> None:
    """
    Uloží výsledky do CSV souboru.

    Args:
        X: Dataset
        labels: Výsledky clusteringu
        filename: Název výstupního souboru
    """
    X_np = np.array(X)
    df = pd.DataFrame()

    # Přidání dimenzí
    for i in range(X_np.shape[1]):
        df[f'dim_{i + 1}'] = X_np[:, i]

    df['cluster'] = labels
    df.to_csv(filename, index=False)
    print(f"Výsledky byly uloženy do {filename}")


def main():
    # Nastavení seed pro reprodukovatelnost
    np.random.seed(42)

    # Test pro 2D data
    print("Test 2D dat:")
    print("-" * 50)

    # Generování 2D dat
    print("Generování 2D dat...")
    X_2d = generate_test_data(n_points=300, n_dims=2)

    # Vytvoření a spuštění DBSCAN
    print("Spouštění DBSCAN...")
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    labels_2d = dbscan.fit(X_2d)

    # Výpis základních statistik
    unique_labels = set(labels_2d)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = labels_2d.count(-1)

    print(f"Počet nalezených clusterů: {n_clusters}")
    print(f"Počet bodů označených jako šum: {n_noise}")

    # Vizualizace výsledků
    print("Vizualizace 2D výsledků...")
    visualize_clusters(X_2d, labels_2d)

    # Uložení výsledků
    save_results(X_2d, labels_2d, 'dbscan_results_2d.csv')

    # Test pro 3D data
    print("\nTest 3D dat:")
    print("-" * 50)

    # Generování 3D dat
    print("Generování 3D dat...")
    X_3d = generate_test_data(n_points=300, n_dims=3)

    # DBSCAN pro 3D data
    print("Spouštění DBSCAN...")
    labels_3d = dbscan.fit(X_3d)

    # Výpis základních statistik
    unique_labels = set(labels_3d)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = labels_3d.count(-1)

    print(f"Počet nalezených clusterů: {n_clusters}")
    print(f"Počet bodů označených jako šum: {n_noise}")

    # Vizualizace 3D výsledků
    print("Vizualizace 3D výsledků...")
    visualize_clusters(X_3d, labels_3d)

    # Uložení 3D výsledků
    save_results(X_3d, labels_3d, 'dbscan_results_3d.csv')


if __name__ == "__main__":
    main()