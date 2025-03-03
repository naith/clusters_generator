import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_blobs
from lib.mean_shift import MeanShift


def test_mean_shift():
    # Generování 2D dat
    X_2D, _ = make_moons(n_samples=300, noise=0.05)
    ms = MeanShift(bandwidth=0.5)
    ms.fit(X_2D)

    # Plotly vizualizace 2D
    plot_2d_results(X_2D, ms.labels_, ms.cluster_centers_, title="Mean Shift - 2D (Syntetická data)")

    # Generování 3D dat
    X_3D, _ = make_blobs(n_samples=300, centers=3, n_features=3, random_state=42)
    ms_3d = MeanShift(bandwidth=2.0)
    ms_3d.fit(X_3D)

    # Plotly 3D vizualizace
    plot_3d_results(X_3D, ms_3d.labels_, ms_3d.cluster_centers_, title="Mean Shift - 3D (Syntetická data)")


def plot_2d_results(X, labels, centers, title="Mean Shift - 2D"):
    """
    Vizualizuje výsledky Mean Shift clusteringu pro 2D data pomocí Plotly.

    Args:
        X: Data (n_samples, 2)
        labels: Výsledné clustery
        centers: Centra clusterů
        title: Titulek grafu
    """
    # Vytvoříme scatter plot s body barevně odlišenými podle clusterů
    fig = px.scatter(
        x=X[:, 0],
        y=X[:, 1],
        color=[f'Cluster {l}' for l in labels],
        title=title,
        labels={'x': 'X', 'y': 'Y', 'color': 'Cluster'},
        width=900,
        height=700
    )

    # Zmenšíme velikost bodů
    fig.update_traces(marker=dict(size=4))

    # Přidáme centra clusterů
    fig.add_trace(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='x',
                line=dict(width=2)
            ),
            name='Cluster Centers'
        )
    )

    fig.show()


def plot_3d_results(X, labels, centers, title="Mean Shift - 3D"):
    """
    Vizualizuje výsledky Mean Shift clusteringu pro 3D data pomocí Plotly.

    Args:
        X: Data (n_samples, 3)
        labels: Výsledné clustery
        centers: Centra clusterů
        title: Titulek grafu
    """
    # Vytvoříme 3D scatter plot s body barevně odlišenými podle clusterů
    fig = px.scatter_3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        color=[f'Cluster {l}' for l in labels],
        title=title,
        labels={'x': 'X', 'y': 'Y', 'z': 'Z', 'color': 'Cluster'},
        width=900,
        height=700
    )

    # Zmenšíme velikost bodů
    fig.update_traces(marker=dict(size=3))

    # Přidáme centra clusterů
    fig.add_trace(
        go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode='markers',
            marker=dict(
                color='red',
                size=6,
                symbol='cross'
            ),
            name='Cluster Centers'
        )
    )

    # Nastavíme vhodný pohled kamery
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
        data: Numpy array pro Mean Shift clustering
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


def save_results(X, labels, centers, filename='mean_shift_results.csv'):
    """
    Uloží výsledky do CSV souboru.

    Args:
        X: Data použitá pro clustering
        labels: Výsledky clusteringu
        centers: Centra clusterů
        filename: Název výstupního souboru
    """
    try:
        # Vytvoření DataFramu
        df = pd.DataFrame()

        # Přidání dimenzí
        for i in range(X.shape[1]):
            df[f'dim_{i + 1}'] = X[:, i]

        # Přidání výsledků clusteringu
        df['point_type'] = ['cluster' for _ in labels]
        df['cluster_id'] = labels

        # Uložení do CSV
        df.to_csv(filename, index=False)
        print(f"Výsledky byly uloženy do {filename}")

        # Uložíme také centra
        centers_df = pd.DataFrame()
        for i in range(centers.shape[1]):
            centers_df[f'dim_{i + 1}'] = centers[:, i]
        centers_df['point_type'] = ['center' for _ in range(centers.shape[0])]
        centers_df['cluster_id'] = range(centers.shape[0])

        centers_filename = filename.replace('.csv', '_centers.csv')
        centers_df.to_csv(centers_filename, index=False)
        print(f"Centra clusterů byla uložena do {centers_filename}")

    except Exception as e:
        print(f"Chyba při ukládání výsledků: {e}")


def process_csv_file(csv_file_path, bandwidth=0.5):
    """
    Načte data z CSV, spustí Mean Shift clustering a vizualizuje výsledky.

    Args:
        csv_file_path: Cesta k CSV souboru
        bandwidth: Šířka pásma pro Mean Shift
    """
    # Načtení dat z CSV
    X = load_csv_data(csv_file_path)

    if X is None:
        print("Nepodařilo se načíst data z CSV souboru.")
        return

    # Spuštění Mean Shift
    print(f"Spouštím Mean Shift s bandwidth={bandwidth}...")
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(X)

    print(f"Nalezeno {len(np.unique(ms.labels_))} clusterů.")

    # Vizualizace podle počtu dimenzí
    if X.shape[1] == 2:
        plot_2d_results(X, ms.labels_, ms.cluster_centers_,
                        title=f"Mean Shift - 2D z CSV (bandwidth={bandwidth})")
    elif X.shape[1] == 3:
        plot_3d_results(X, ms.labels_, ms.cluster_centers_,
                        title=f"Mean Shift - 3D z CSV (bandwidth={bandwidth})")
    else:
        print(f"Vizualizace není podporována pro {X.shape[1]} dimenzí.")

    # Uložení výsledků
    output_filename = csv_file_path.replace('.csv', f'_mean_shift_bw{bandwidth}.csv')
    save_results(X, ms.labels_, ms.cluster_centers_, output_filename)


if __name__ == "__main__":
    # Pro použití původního testování na syntetických datech:
    # test_mean_shift()

    # Pro zpracování CSV souboru:
    # Odkomentujte následující řádky a upravte cestu k souboru a parametry
    csv_file_path = "data/shapes/shapes_dataset.csv"  # Upravte podle potřeby

    # Nastavení bandwidth - klíčový parametr Mean Shift algoritmu
    # Menší hodnota = více detailů, více clusterů
    # Větší hodnota = méně clusterů, více generalizace
    # Upravte podle charakteru vašich dat
    bandwidth = 0.5

    process_csv_file(csv_file_path, bandwidth=bandwidth)