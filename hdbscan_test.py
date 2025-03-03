import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    Jednoduchý scatter plot 2D dat podle cluster labelů s legendou pomocí Plotly.
    """
    # Vytvoříme DataFrame pro Plotly
    df = pd.DataFrame({
        'x': [p[0] for p in data],
        'y': [p[1] for p in data],
        'cluster': [f'Cluster {l}' if l >= 0 else 'Noise' for l in labels]
    })

    # Použijeme Plotly Express pro vytvoření interaktivního scatter plotu
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        title=title,
        labels={'x': 'X', 'y': 'Y', 'cluster': 'Cluster'},
        width=1200,
        height=800
    )

    # Nastavíme menší velikost bodů
    fig.update_traces(marker=dict(size=5))

    # Upravíme barevnou škálu tak, aby šum (Noise) byl černý
    fig.update_traces(
        selector=dict(name='Noise'),
        marker=dict(color='black')
    )

    # Zobrazíme graf
    fig.show()


def plot_3d(data, labels, title="HDBSCAN (3D)"):
    """
    Jednoduchý scatter plot 3D dat podle cluster labelů s legendou pomocí Plotly.
    """
    # Vytvoříme DataFrame pro Plotly
    df = pd.DataFrame({
        'x': [p[0] for p in data],
        'y': [p[1] for p in data],
        'z': [p[2] for p in data],
        'cluster': [f'Cluster {l}' if l >= 0 else 'Noise' for l in labels]
    })

    # Použijeme Plotly Express pro vytvoření interaktivního 3D scatter plotu
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='cluster',
        title=title,
        labels={'x': 'X', 'y': 'Y', 'z': 'Z', 'cluster': 'Cluster'},
        width=1200,
        height=800
    )

    # Nastavíme menší velikost bodů
    fig.update_traces(marker=dict(size=3))

    # Upravíme barevnou škálu tak, aby šum (Noise) byl černý
    fig.update_traces(
        selector=dict(name='Noise'),
        marker=dict(color='black')
    )

    # Nastavíme vhodný pohled kamery
    fig.update_layout(scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5)
    ))

    # Zobrazíme graf
    fig.show()


def load_csv_data(csv_file_path):
    """
    Načte data z CSV souboru a ignoruje poslední dva sloupce.

    Args:
        csv_file_path: Cesta k CSV souboru

    Returns:
        data: List bodů (n-tic) pro HDBSCAN
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

        # Konverze na list n-tic
        data = [tuple(row) for row in X_df.values.astype(float)]
        print(f"Načteno {len(data)} bodů.")

        return data

    except Exception as e:
        print(f"Chyba při načítání CSV souboru: {e}")
        return None


def save_results(data, labels, filename='hdbscan_results.csv'):
    """
    Uloží výsledky do CSV souboru ve stejném formátu jako původní CSV.

    Args:
        data: List bodů (n-tic)
        labels: Výsledky clusteringu
        filename: Název výstupního souboru
    """
    try:
        # Vytvoření DataFramu
        df = pd.DataFrame()

        # Přidání dimenzí
        dim = len(data[0])
        for i in range(dim):
            df[f'dim_{i + 1}'] = [point[i] for point in data]

        # Přidání výsledků clusteringu
        df['point_type'] = ['noise' if label == -1 else 'cluster' for label in labels]
        df['cluster_id'] = labels

        # Uložení do CSV
        df.to_csv(filename, index=False)
        print(f"Výsledky byly uloženy do {filename}")

    except Exception as e:
        print(f"Chyba při ukládání výsledků: {e}")


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


def process_csv_file(csv_file_path, min_points=5, min_cluster_size=5):
    """
    Načte data z CSV, spustí HDBSCAN a vizualizuje výsledky.

    Args:
        csv_file_path: Cesta k CSV souboru
        min_points: Parametr min_points pro HDBSCAN
        min_cluster_size: Parametr min_cluster_size pro HDBSCAN
    """
    # Načtení dat z CSV
    data = load_csv_data(csv_file_path)

    if data is None:
        print("Nepodařilo se načíst data z CSV souboru.")
        return

    # Zjištění počtu dimenzí
    dimensions = len(data[0])
    print(f"Data mají {dimensions} dimenzí.")

    # Spuštění HDBSCAN
    print(f"Spouštím HDBSCAN s parametry min_points={min_points}, min_cluster_size={min_cluster_size}...")
    labels = hdbscan(data, min_points=min_points, min_cluster_size=min_cluster_size)

    # Výpis statistik
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = labels.count(-1)

    print(f"Počet nalezených clusterů: {n_clusters}")
    print(f"Počet bodů označených jako šum: {n_noise}")

    # Vizualizace podle počtu dimenzí
    if dimensions == 2:
        plot_2d(data, labels, title=f"HDBSCAN výsledek z CSV (2D)")
    elif dimensions == 3:
        plot_3d(data, labels, title=f"HDBSCAN výsledek z CSV (3D)")
    else:
        print(f"Vizualizace není podporována pro {dimensions} dimenzí.")

    # Uložení výsledků
    output_filename = csv_file_path.replace('.csv', '_hdbscan_results.csv')
    save_results(data, labels, output_filename)


if __name__ == "__main__":
    # Pro použití původního testování:
    # main()

    # Pro zpracování CSV souboru:
    # Odkomentujte následující řádky a upravte cestu k souboru a parametry
    csv_file_path = "data/shapes/shapes_dataset.csv"  # Upravte podle potřeby
    process_csv_file(csv_file_path, min_points=5, min_cluster_size=5)