import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.datasets import make_moons, make_blobs, make_circles
import argparse
import random
import time

# Import jednotlivých algoritmů
from lib.k_nn import KNN
from lib.dbscan import DBSCAN
from lib.hdbscan import hdbscan
from lib.spectral_auto import spectral_clustering
from lib.mean_shift import MeanShift
from lib.optics import OPTICS

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Nastavení seed pro reprodukovatelnost výsledků
np.random.seed(42)
random.seed(42)


# Přidejte tento import na začátek souboru

def evaluate_clustering(X, labels):
    """
    Vyhodnotí kvalitu clusteringu pomocí různých metrik.

    Args:
        X: Data použitá pro clustering
        labels: Výsledky clusteringu (přiřazení do clusterů)

    Returns:
        tuple: (silhouette, calinski_harabasz, davies_bouldin)
    """
    # Zkontrolujeme, zda máme více než jeden cluster a žádné body nejsou označeny jako šum
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l != -1])

    # Příprava dat - odstranění bodů označených jako šum (-1)
    if -1 in labels:
        mask = np.array(labels) != -1
        X_eval = X[mask]
        labels_eval = np.array(labels)[mask]
    else:
        X_eval = X
        labels_eval = labels

    # Pokud máme alespoň 2 clustery a alespoň 1 bod v každém clusteru
    if n_clusters >= 2 and len(np.unique(labels_eval)) >= 2 and len(X_eval) > n_clusters:
        try:
            # Silhouette score - vyšší je lepší (-1 až 1)
            silhouette = silhouette_score(X_eval, labels_eval)

            # Calinski-Harabasz index - vyšší je lepší
            calinski_harabasz = calinski_harabasz_score(X_eval, labels_eval)

            # Davies-Bouldin index - nižší je lepší
            davies_bouldin = davies_bouldin_score(X_eval, labels_eval)

            print("\nVýsledky evaluace clusteringu:")
            print(f"  Počet clusterů: {n_clusters}")
            print(f"  Silhouette score: {silhouette:.4f} (vyšší je lepší, rozsah: -1 až 1)")
            print(f"  Calinski-Harabasz index: {calinski_harabasz:.4f} (vyšší je lepší)")
            print(f"  Davies-Bouldin index: {davies_bouldin:.4f} (nižší je lepší)")

            return silhouette, calinski_harabasz, davies_bouldin
        except Exception as e:
            print(f"\nChyba při výpočtu metrik: {e}")
    else:
        print(
            f"\nNení možné vypočítat metriky: Nedostatek clusterů nebo bodů ({n_clusters} clusterů, {len(X_eval)} bodů).")

    return None, None, None


def generate_synthetic_data(dataset_type="moons", n_samples=300, noise=0.1, n_dims=2):
    """
    Generuje syntetická data pro testování shlukovacích algoritmů.

    Args:
        dataset_type: Typ datasetu ("moons", "blobs", "circles")
        n_samples: Počet vzorků
        noise: Úroveň šumu
        n_dims: Počet dimenzí (pro typ "blobs")

    Returns:
        X: Numpy array s daty
    """
    print(f"Generuji syntetická data: {dataset_type}, {n_samples} vzorků, šum {noise}")

    if dataset_type == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=noise)
    elif dataset_type == "blobs":
        X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=n_dims, random_state=42)
    elif dataset_type == "circles":
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    else:
        raise ValueError(f"Neznámý typ datasetu: {dataset_type}")

    return X


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


def save_results_to_csv(X, labels, centers=None, algorithm="clustering", filename=None):
    """
    Uloží výsledky do CSV souboru.

    Args:
        X: Data použitá pro clustering
        labels: Výsledky clusteringu
        centers: Centra clusterů (volitelné)
        algorithm: Název algoritmu
        filename: Název výstupního souboru (volitelné)
    """
    try:
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"results_{algorithm}_{timestamp}.csv"

        # Vytvoření DataFramu
        df = pd.DataFrame()

        # Přidání dimenzí
        for i in range(X.shape[1]):
            df[f'dim_{i + 1}'] = X[:, i]

        # Přidání výsledků clusteringu
        if algorithm in ["dbscan", "hdbscan"]:
            df['point_type'] = ['noise' if l == -1 else 'cluster' for l in labels]
        else:
            df['point_type'] = ['cluster' for _ in labels]

        df['cluster_id'] = labels

        # Vytvoření adresáře pro výsledky, pokud neexistuje
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

        # Uložení do CSV
        df.to_csv(filename, index=False)
        print(f"Výsledky byly uloženy do {filename}")

        # Pokud máme centra, uložíme je také
        if centers is not None:
            centers_df = pd.DataFrame()
            for i in range(centers.shape[1]):
                centers_df[f'dim_{i + 1}'] = centers[:, i]
            centers_df['point_type'] = ['center' for _ in range(centers.shape[0])]
            centers_df['cluster_id'] = range(centers.shape[0])

            centers_filename = filename.replace('.csv', '_centers.csv')
            centers_df.to_csv(centers_filename, index=False)
            print(f"Centra clusterů byla uložena do {centers_filename}")

        return filename

    except Exception as e:
        print(f"Chyba při ukládání výsledků: {e}")
        return None


def plot_2d(X, labels, centers=None, title="Clustering Results (2D)", show_plot=True, save_plot=False, filename=None):
    """
    Vizualizuje výsledky clusteringu pro 2D data pomocí Plotly.

    Args:
        X: Data (n_samples, 2)
        labels: Výsledné clustery
        centers: Centra clusterů (volitelné)
        title: Titulek grafu
        show_plot: Zda zobrazit graf
        save_plot: Zda uložit graf
        filename: Název souboru pro uložení (volitelné)
    """
    # Připravíme data pro vizualizaci
    is_noise = np.array(labels) == -1 if -1 in labels else np.zeros(len(labels), dtype=bool)
    cluster_labels = [f'Noise' if is_n else f'Cluster {l}' for is_n, l in zip(is_noise, labels)]

    # Vytvoříme dataframe pro Plotly
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'cluster': cluster_labels
    })

    # Vytvoříme scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        title=title,
        labels={'x': 'X', 'y': 'Y', 'cluster': 'Cluster'},
        width=900,
        height=700
    )

    # Zajistíme, aby šum byl černý
    if -1 in labels:
        fig.update_traces(
            selector=dict(name='Noise'),
            marker=dict(color='black')
        )

    # Zmenšíme velikost bodů
    fig.update_traces(marker=dict(size=4))

    # Přidáme centra clusterů, pokud jsou k dispozici
    if centers is not None:
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

    # Zobrazení nebo uložení grafu
    if save_plot:
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"plot_2d_{timestamp}.html"

        # Vytvoření adresáře pro grafy, pokud neexistuje
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

        fig.write_html(filename)
        print(f"Graf byl uložen do {filename}")

    if show_plot:
        fig.show()


def plot_3d(X, labels, centers=None, title="Clustering Results (3D)", show_plot=True, save_plot=False, filename=None):
    """
    Vizualizuje výsledky clusteringu pro 3D data pomocí Plotly.

    Args:
        X: Data (n_samples, 3)
        labels: Výsledné clustery
        centers: Centra clusterů (volitelné)
        title: Titulek grafu
        show_plot: Zda zobrazit graf
        save_plot: Zda uložit graf
        filename: Název souboru pro uložení (volitelné)
    """
    # Připravíme data pro vizualizaci
    is_noise = np.array(labels) == -1 if -1 in labels else np.zeros(len(labels), dtype=bool)
    cluster_labels = [f'Noise' if is_n else f'Cluster {l}' for is_n, l in zip(is_noise, labels)]

    # Vytvoříme dataframe pro Plotly
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'z': X[:, 2],
        'cluster': cluster_labels
    })

    # Vytvoříme 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='cluster',
        title=title,
        labels={'x': 'X', 'y': 'Y', 'z': 'Z', 'cluster': 'Cluster'},
        width=900,
        height=700
    )

    # Zajistíme, aby šum byl černý
    if -1 in labels:
        fig.update_traces(
            selector=dict(name='Noise'),
            marker=dict(color='black')
        )

    # Zmenšíme velikost bodů
    fig.update_traces(marker=dict(size=3))

    # Přidáme centra clusterů, pokud jsou k dispozici
    if centers is not None:
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

    # Zobrazení nebo uložení grafu
    if save_plot:
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"plot_3d_{timestamp}.html"

        # Vytvoření adresáře pro grafy, pokud neexistuje
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

        fig.write_html(filename)
        print(f"Graf byl uložen do {filename}")

    if show_plot:
        fig.show()


def run_optics(X, eps=0.5, min_samples=5, eps_dbscan=0.5, show_plot=True, save_plot=False, save_results=False,
               output_dir="results/optics"):
    """
    Spustí OPTICS algoritmus na datech X.

    Args:
        X: Data pro clustering
        eps: Parametr epsilon (max vzdálenost pro sousedy)
        min_samples: Minimální počet bodů v sousedství
        eps_dbscan: Parametr epsilon pro extrakci clusterů jako v DBSCANu
        show_plot: Zda zobrazit graf
        save_plot: Zda uložit graf
        save_results: Zda uložit výsledky
        output_dir: Adresář pro výstupy
    """
    print(f"\n--- OPTICS (eps={eps}, min_samples={min_samples}, eps_dbscan={eps_dbscan}) ---")

    # Inicializace a spuštění OPTICS
    optics = OPTICS(eps=eps, min_samples=min_samples)
    optics.fit(X)

    # Extrakce clusterů pomocí metody podobné DBSCANu
    labels = optics.extract_dbscan(eps_dbscan=eps_dbscan)

    # Výpis základních statistik
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = np.sum(labels == -1)

    print(f"Počet nalezených clusterů: {n_clusters}")
    print(f"Počet bodů označených jako šum: {n_noise}")

    # Vizualizace výsledků
    title = f"OPTICS Clustering (eps={eps}, min_samples={min_samples}, eps_dbscan={eps_dbscan})"
    if X.shape[1] == 2:
        plot_2d(
            X, labels, centers=None, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/optics_eps{eps}_ms{min_samples}_epsdb{eps_dbscan}_2d.html" if save_plot else None
        )
    elif X.shape[1] == 3:
        plot_3d(
            X, labels, centers=None, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/optics_eps{eps}_ms{min_samples}_epsdb{eps_dbscan}_3d.html" if save_plot else None
        )
    else:
        print(f"Vizualizace není podporována pro {X.shape[1]} dimenzí.")

    # Uložení výsledků
    if save_results:
        save_results_to_csv(X, labels, centers=None, algorithm="optics",
                            filename=f"{output_dir}/optics_eps{eps}_ms{min_samples}_epsdb{eps_dbscan}_results.csv")

    evaluate_clustering(X, labels)
    return labels


def run_knn(X, k=2, show_plot=True, save_plot=False, save_results=False, output_dir="results/knn", labels=None):
    """
    Spustí KNN algoritmus na datech X.

    Args:
        X: Data pro clustering
        k: Počet sousedů
        show_plot: Zda zobrazit graf
        save_plot: Zda uložit graf
        save_results: Zda uložit výsledky
        output_dir: Adresář pro výstupy
    """
    print(f"\n--- KNN (k={k}) ---")

    # V KNN potřebujeme trénovací data a labely
    # Pro ukázku použijeme jednoduché labely (všechny body mají stejnou třídu)
    y_train = np.zeros(len(X))

    # Inicializace a spuštění KNN
    knn = KNN(k=k)
    y_pred = knn.predict(X, y_train, X)

    # Vizualizace výsledků
    title = f"KNN Clustering (k={k})"
    if X.shape[1] == 2:
        plot_2d(
            X, y_pred, centers=None, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/knn_k{k}_2d.html" if save_plot else None
        )
    elif X.shape[1] == 3:
        plot_3d(
            X, y_pred, centers=None, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/knn_k{k}_3d.html" if save_plot else None
        )
    else:
        print(f"Vizualizace není podporována pro {X.shape[1]} dimenzí.")

    # Uložení výsledků
    if save_results:
        save_results_to_csv(X, y_pred, centers=None, algorithm="knn",
                            filename=f"{output_dir}/knn_k{k}_results.csv")

    evaluate_clustering(X, labels)
    return y_pred


def run_dbscan(X, eps=0.5, min_samples=5, show_plot=True, save_plot=False, save_results=False, output_dir="results/dbscan"):
    """
    Spustí DBSCAN algoritmus na datech X.

    Args:
        X: Data pro clustering
        eps: Parametr epsilon (max vzdálenost pro sousedy)
        min_samples: Minimální počet bodů v sousedství
        show_plot: Zda zobrazit graf
        save_plot: Zda uložit graf
        save_results: Zda uložit výsledky
        output_dir: Adresář pro výstupy
    """
    print(f"\n--- DBSCAN (eps={eps}, min_samples={min_samples}) ---")

    # Inicializace a spuštění DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit(X)

    # Výpis základních statistik
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = np.count_nonzero(labels == -1)


    print(f"Počet nalezených clusterů: {n_clusters}")
    print(f"Počet bodů označených jako šum: {n_noise}")

    # Vizualizace výsledků
    title = f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})"
    if X.shape[1] == 2:
        plot_2d(
            X, labels, centers=None, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/dbscan_eps{eps}_ms{min_samples}_2d.html" if save_plot else None
        )
    elif X.shape[1] == 3:
        plot_3d(
            X, labels, centers=None, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/dbscan_eps{eps}_ms{min_samples}_3d.html" if save_plot else None
        )
    else:
        print(f"Vizualizace není podporována pro {X.shape[1]} dimenzí.")

    # Uložení výsledků
    if save_results:
        save_results_to_csv(X, labels, centers=None, algorithm="dbscan",
                            filename=f"{output_dir}/dbscan_eps{eps}_ms{min_samples}_results.csv")

    evaluate_clustering(X, labels)
    return labels


def run_hdbscan(X, min_points=5, min_cluster_size=5, show_plot=True, save_plot=False, save_results=False,output_dir="results/hdbscan"):
    """
    Spustí HDBSCAN algoritmus na datech X.

    Args:
        X: Data pro clustering
        min_points: Minimální počet bodů pro výpočet core distance
        min_cluster_size: Minimální velikost clusteru
        show_plot: Zda zobrazit graf
        save_plot: Zda uložit graf
        save_results: Zda uložit výsledky
        output_dir: Adresář pro výstupy
    """
    print(f"\n--- HDBSCAN (min_points={min_points}, min_cluster_size={min_cluster_size}) ---")

    # Konverze numpy array na list of tuples, pokud X je numpy array
    if isinstance(X, np.ndarray):
        X_list = [tuple(x) for x in X]
    else:
        X_list = X

    # Spuštění HDBSCAN
    labels = hdbscan(X_list, min_points=min_points, min_cluster_size=min_cluster_size)

    # Výpis základních statistik
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = labels.count(-1)

    print(f"Počet nalezených clusterů: {n_clusters}")
    print(f"Počet bodů označených jako šum: {n_noise}")

    # Vizualizace výsledků
    title = f"HDBSCAN Clustering (min_points={min_points}, min_cluster_size={min_cluster_size})"
    if X.shape[1] == 2:
        plot_2d(
            X, labels, centers=None, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/hdbscan_mp{min_points}_mcs{min_cluster_size}_2d.html" if save_plot else None
        )
    elif X.shape[1] == 3:
        plot_3d(
            X, labels, centers=None, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/hdbscan_mp{min_points}_mcs{min_cluster_size}_3d.html" if save_plot else None
        )
    else:
        print(f"Vizualizace není podporována pro {X.shape[1]} dimenzí.")

    # Uložení výsledků
    if save_results:
        save_results_to_csv(X, labels, centers=None, algorithm="hdbscan",
                            filename=f"{output_dir}/hdbscan_mp{min_points}_mcs{min_cluster_size}_results.csv")

    evaluate_clustering(X, labels)
    return labels


def run_spectral(X, k=2, sigma=0.5, use_knn=True, knn_k=10, normalized=True,
                 show_plot=True, save_plot=True, save_results=False, output_dir="results/spectral"):
    """
    Spustí Spectral Clustering algoritmus na datech X.

    Args:
        X: Data pro clustering
        k: Počet shluků
        sigma: Parametr šířky jádra
        use_knn: Použít k-NN graf místo plného grafu
        knn_k: Počet sousedů pro k-NN
        normalized: Použít normalizovaný Laplacián
        show_plot: Zda zobrazit graf
        save_plot: Zda uložit graf
        save_results: Zda uložit výsledky
        output_dir: Adresář pro výstupy
    """
    print(f"\n--- Spectral Clustering (k={k}, sigma={sigma}, use_knn={use_knn}, knn_k={knn_k}) ---")

    # Spuštění Spectral Clustering
    labels = spectral_clustering(X, k=k, sigma=sigma, use_knn=use_knn, knn_k=knn_k, normalized=normalized)

    # Výpis základních statistik
    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    print(f"Počet nalezených clusterů: {n_clusters}")

    # Vizualizace výsledků
    title = f"Spectral Clustering (k={k}, sigma={sigma})"
    if X.shape[1] == 2:
        plot_2d(
            X, labels, centers=None, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/spectral_k{k}_sigma{sigma}_2d.html" if save_plot else None
        )
    elif X.shape[1] == 3:
        plot_3d(
            X, labels, centers=None, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/spectral_k{k}_sigma{sigma}_3d.html" if save_plot else None
        )
    else:
        print(f"Vizualizace není podporována pro {X.shape[1]} dimenzí.")

    # Uložení výsledků
    if save_results:
        save_results_to_csv(X, labels, centers=None, algorithm="spectral",
                            filename=f"{output_dir}/spectral_k{k}_sigma{sigma}_results.csv")

    evaluate_clustering(X, labels)
    return labels


def run_mean_shift(X, bandwidth=0.5, show_plot=True, save_plot=False, save_results=False, output_dir="results/mean_shift"):
    """
    Spustí Mean Shift algoritmus na datech X.

    Args:
        X: Data pro clustering
        bandwidth: Šířka pásma
        show_plot: Zda zobrazit graf
        save_plot: Zda uložit graf
        save_results: Zda uložit výsledky
        output_dir: Adresář pro výstupy
    """
    print(f"\n--- Mean Shift (bandwidth={bandwidth}) ---")

    # Inicializace a spuštění Mean Shift
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(X)

    # Získání výsledků
    labels = ms.labels_
    centers = ms.cluster_centers_

    # Výpis základních statistik
    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    print(f"Počet nalezených clusterů: {n_clusters}")

    # Vizualizace výsledků
    title = f"Mean Shift Clustering (bandwidth={bandwidth})"
    if X.shape[1] == 2:
        plot_2d(
            X, labels, centers=centers, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/mean_shift_bw{bandwidth}_2d.html" if save_plot else None
        )
    elif X.shape[1] == 3:
        plot_3d(
            X, labels, centers=centers, title=title,
            show_plot=show_plot, save_plot=save_plot,
            filename=f"{output_dir}/mean_shift_bw{bandwidth}_3d.html" if save_plot else None
        )
    else:
        print(f"Vizualizace není podporována pro {X.shape[1]} dimenzí.")

    # Uložení výsledků
    if save_results:
        save_results_to_csv(X, labels, centers=centers, algorithm="mean_shift",
                            filename=f"{output_dir}/mean_shift_bw{bandwidth}_results.csv")

    evaluate_clustering(X, labels)
    return labels, centers


def run_all_algorithms(X, show_plot=True, save_plot=False, save_results=False, output_dir="results/all"):
    """
    Spustí všechny shlukovací algoritmy na datech X.

    Args:
        X: Data pro clustering
        show_plot: Zda zobrazit grafy
        save_plot: Zda uložit grafy
        save_results: Zda uložit výsledky
        output_dir: Adresář pro výstupy
    """
    print(f"\n=== Spouštím všechny algoritmy na datech ===")

    # Vytvoření výstupního adresáře
    os.makedirs(output_dir, exist_ok=True)

    # Spuštění všech algoritmů
    # run_knn(X, k=5, show_plot=show_plot, save_plot=save_plot, save_results=save_results, output_dir=output_dir)
    run_dbscan(X, eps=1.2, min_samples=5, show_plot=show_plot, save_plot=save_plot,
               save_results=should_save_results, output_dir=output_dir)
    run_hdbscan(X, min_points=5, min_cluster_size=5, show_plot=show_plot, save_plot=save_plot,
                save_results=should_save_results, output_dir=output_dir)
    # run_mean_shift(X, bandwidth=0.5, show_plot=show_plot, save_plot=save_plot, save_results=save_results,
    #               output_dir=output_dir)
    run_optics(X, eps=1, min_samples=5, eps_dbscan=1, show_plot=show_plot, save_plot=save_plot,
               save_results=should_save_results, output_dir=output_dir)

    run_spectral(X, k=9, sigma=1, show_plot=show_plot, save_plot=save_plot, save_results=should_save_results,
                 output_dir=output_dir, use_knn=True, normalized=False)

    print("\n=== Dokončeno spuštění všech algoritmů ===")


# Finally, modify the main part to include OPTICS as an option
if __name__ == "__main__":
    # Nastavení vstupních parametrů - pro použití přímo v kódu
    # Změňte tyto parametry podle potřeby

    # 1. Vstupní data - CSV nebo syntetická
    input_type = "csv"  # "synthetic" nebo "csv"
    csv_file_path = "data/sphere/sphere_cluster_dataset.csv"  # Cesta k CSV souboru
    # csv_file_path = "data/string/string_cluster_dataset.csv"  # Cesta k CSV souboru
    # csv_file_path = "data/rings/ring_cluster_dataset.csv"  # Cesta k CSV souboru
    # csv_file_path = "data/shapes/shapes_dataset.csv"  # Cesta k CSV souboru

    # 2. Parametry pro syntetická data
    synthetic_type = "moons"  # "moons", "blobs" nebo "circles"
    n_samples = 300
    noise = 0.1
    n_dims = 3  # Pro typ "blobs"

    # 3. Parametry pro algoritmy
    algorithm = "all"  # "all", "knn", "dbscan", "hdbscan", "spectral", "mean_shift" nebo "optics"

    # 4. Výstupní parametry
    show_plot = True  # Zda zobrazit grafy
    save_plot = True  # Zda uložit grafy
    should_save_results = True  # Zda uložit výsledky clusteringu
    output_dir = "results"  # Adresář pro výstupy

    # Načtení nebo generování dat
    if input_type == "csv":
        X = load_csv_data(csv_file_path)
        if X is None:
            print("Nepodařilo se načíst data z CSV souboru. Ukončuji program.")
            exit(1)
    else:
        X = generate_synthetic_data(dataset_type=synthetic_type, n_samples=n_samples, noise=noise, n_dims=n_dims)

    # Spuštění zvoleného algoritmu nebo všech algoritmů
    if algorithm == "all":
        run_all_algorithms(X, show_plot=show_plot, save_plot=save_plot, save_results=should_save_results,
                           output_dir=output_dir)
    elif algorithm == "knn":
        run_knn(X, k=2, show_plot=show_plot, save_plot=save_plot, save_results=should_save_results,
                output_dir=output_dir)
    elif algorithm == "dbscan":
        run_dbscan(X, eps=1.2, min_samples=5, show_plot=show_plot, save_plot=save_plot,
                   save_results=should_save_results, output_dir=output_dir)
    elif algorithm == "hdbscan":
        run_hdbscan(X, min_points=3, min_cluster_size=2, show_plot=show_plot, save_plot=save_plot,
                    save_results=should_save_results, output_dir=output_dir)
    elif algorithm == "spectral":
        run_spectral(X, k=7, sigma=0, show_plot=show_plot, save_plot=save_plot, save_results=should_save_results,
                     output_dir=output_dir, use_knn=True, normalized=False)
    elif algorithm == "mean_shift":
        run_mean_shift(X, bandwidth=0.5, show_plot=show_plot, save_plot=save_plot, save_results=should_save_results,
                       output_dir=output_dir)
    elif algorithm == "optics":
        run_optics(X, eps=1, min_samples=5, eps_dbscan=1, show_plot=show_plot, save_plot=save_plot,
                   save_results=should_save_results, output_dir=output_dir)
    else:
        print(f"Neznámý algoritmus: {algorithm}")
