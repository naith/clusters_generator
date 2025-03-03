import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lib.spectral import spectral_clustering


def generate_two_moons_data(n_samples=300, noise=0.06):
    """
    Generuje 2D 'two moons' dataset.
    """
    np.random.seed(789)
    n_half = n_samples // 2

    theta1 = np.linspace(0, np.pi, n_half)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)
    data1 = np.vstack((x1, y1)).T

    theta2 = np.linspace(0, np.pi, n_half)
    x2 = 1 - np.cos(theta2)
    y2 = -np.sin(theta2) + 0.5
    data2 = np.vstack((x2, y2)).T

    data1 += noise * np.random.randn(n_half, 2)
    data2 += noise * np.random.randn(n_half, 2)

    X = np.vstack([data1, data2])
    return X


def generate_3d_data():
    """
    Vytvoří jednoduchá 3D data se 3 shluky (Gauss).
    """
    np.random.seed(1234)

    # cluster 1
    X1 = np.random.randn(100, 3) * 0.3 + np.array([0, 0, 0])
    # cluster 2
    X2 = np.random.randn(100, 3) * 0.5 + np.array([3, 3, 1])
    # cluster 3
    X3 = np.random.randn(100, 3) * 0.7 + np.array([-2, 2, 4])

    X = np.vstack([X1, X2, X3])
    return X


def plot_2d(X, labels, title="Spectral Clustering (2D)"):
    """
    Vizualizuje 2D data pomocí Plotly.
    """
    # Vytvoříme DataFrame pro Plotly
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'cluster': [f'Cluster {l}' for l in labels]
    })

    # Vytvoříme interaktivní 2D scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        title=title,
        labels={'x': 'X', 'y': 'Y', 'cluster': 'Cluster'},
        width=800,
        height=800
    )

    # Zmenšíme velikost bodů
    fig.update_traces(marker=dict(size=4))

    # Nastavíme barevnou škálu
    fig.update_layout(coloraxis_colorbar=dict(
        title='Clusters',
    ))

    fig.show()


def plot_3d(X, labels, title="Spectral Clustering (3D)"):
    """
    Vizualizuje 3D data pomocí Plotly.
    """
    # Vytvoříme DataFrame pro Plotly
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'z': X[:, 2],
        'cluster': [f'Cluster {l}' for l in labels]
    })

    # Vytvoříme interaktivní 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='cluster',
        title=title,
        labels={'x': 'X', 'y': 'Y', 'z': 'Z', 'cluster': 'Cluster'},
        width=900,
        height=800
    )

    # Zmenšíme velikost bodů
    fig.update_traces(marker=dict(size=3))

    # Nastavíme lepší pohled kamery
    fig.update_layout(scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5)
    ))

    fig.show()


def load_csv_data(csv_file_path):
    """
    Načte data z CSV souboru a ignoruje poslední dva sloupce.

    Args:
        csv_file_path: Cesta k CSV souboru

    Returns:
        X: Numpy array s daty pro clustering
    """
    try:
        # Načtení dat z CSV
        print(f"Načítám data z: {csv_file_path}")
        df = pd.read_csv(csv_file_path)

        # Kontrola datových typů
        print("Datové typy sloupců:")
        print(df.dtypes)

        # Identifikace sloupců dimenzí
        dimension_cols = [col for col in df.columns if col.startswith('dim_')]

        if not dimension_cols:
            # Pokud nebyly nalezeny sloupce s prefixem 'dim_', použijeme všechny sloupce kromě posledních dvou
            X_df = df.iloc[:, :-2]
            dimension_cols = X_df.columns.tolist()
        else:
            # Jinak použijeme pouze nalezené sloupce dimenzí
            X_df = df[dimension_cols]

        print(f"Používám {len(dimension_cols)} dimenzí pro analýzu, ignoruji poslední dva sloupce.")
        print(f"Dimenze: {dimension_cols}")

        # Konverze na numpy array
        X = X_df.values.astype(float)
        print(f"Načteno {X.shape[0]} bodů s {X.shape[1]} dimenzemi.")

        return X

    except Exception as e:
        print(f"Chyba při načítání CSV souboru: {e}")
        return None


def save_results(X, labels, filename='spectral_results.csv'):
    """
    Uloží výsledky do CSV souboru.

    Args:
        X: Data použitá pro clustering
        labels: Výsledky clusteringu
        filename: Název výstupního souboru
    """
    try:
        # Vytvoření DataFramu
        df = pd.DataFrame()

        # Přidání dimenzí
        for i in range(X.shape[1]):
            df[f'dim_{i + 1}'] = X[:, i]

        # Přidání výsledků clusteringu
        df['point_type'] = ['cluster' for _ in labels]  # V spektrálním clusteringu není šum
        df['cluster_id'] = labels

        # Uložení do CSV
        df.to_csv(filename, index=False)
        print(f"Výsledky byly uloženy do {filename}")

    except Exception as e:
        print(f"Chyba při ukládání výsledků: {e}")


def process_csv_file(csv_file_path, k=2, sigma=0.2, use_knn=True, knn_k=10, normalized=True):
    """
    Načte data z CSV, spustí spektrální clustering a vizualizuje výsledky.

    Args:
        csv_file_path: Cesta k CSV souboru
        k: Počet shluků
        sigma: Parametr šířky jádra
        use_knn: Použít k-NN graf místo plného grafu
        knn_k: Počet sousedů pro k-NN
        normalized: Použít normalizovaný Laplacián
    """
    # Načtení dat z CSV
    X = load_csv_data(csv_file_path)

    if X is None:
        print("Nepodařilo se načíst data z CSV souboru.")
        return

    # Spuštění spektrálního clusteringu
    print(
        f"Spouštím spektrální clustering s parametry k={k}, sigma={sigma}, use_knn={use_knn}, knn_k={knn_k}, normalized={normalized}...")
    labels = spectral_clustering(X, k=k, sigma=sigma, use_knn=use_knn, knn_k=knn_k, normalized=normalized)

    # Vizualizace podle počtu dimenzí
    if X.shape[1] == 2:
        plot_2d(X, labels, title=f"Spectral Clustering z CSV (2D, k={k})")
    elif X.shape[1] == 3:
        plot_3d(X, labels, title=f"Spectral Clustering z CSV (3D, k={k})")
    else:
        print(f"Vizualizace není podporována pro {X.shape[1]} dimenzí.")

    # Uložení výsledků
    output_filename = csv_file_path.replace('.csv', '_spectral_results.csv')
    save_results(X, labels, output_filename)


def main():
    # ---- 1) Two moons (2D) ----
    X_moons = generate_two_moons_data(n_samples=300, noise=0.06)
    labels_moons = spectral_clustering(
        X_moons,
        k=2,
        sigma=0.2,
        use_knn=True,  # k-NN
        knn_k=10,  # 10 sousedů
        normalized=True
    )

    plot_2d(X_moons, labels_moons, title="Spectral Clustering (two moons, 2D)")

    # ---- 2) Gaussovské 3D shluky ----
    X_3d = generate_3d_data()
    labels_3d = spectral_clustering(
        X_3d,
        k=3,
        sigma=1.0,
        use_knn=True,  # i v 3D klidně vyzkoušíme k-NN
        knn_k=20,
        normalized=True
    )

    plot_3d(X_3d, labels_3d, title="Spectral Clustering (3D data)")


if __name__ == "__main__":
    # Pro použití původního testování:
    # main()

    # Pro zpracování CSV souboru:
    # Odkomentujte následující řádky a upravte cestu k souboru a parametry
    csv_file_path = "data/shapes/shapes_dataset.csv"  # Upravte podle potřeby

    # Upravte parametry podle potřeby:
    # k = počet shluků
    # sigma = šířka jádra (větší hodnota = širší sousedství)
    # use_knn = použít k-nejbližších sousedů místo plného grafu (vhodné pro větší datasety)
    # knn_k = počet sousedů pro k-NN
    # normalized = použít normalizovaný Laplacián (obvykle lepší výsledky)
    process_csv_file(csv_file_path, k=2, sigma=0.2, use_knn=True, knn_k=10, normalized=True)