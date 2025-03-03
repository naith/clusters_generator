import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.datasets import make_moons, make_blobs, make_circles
from lib.k_nn import KNN


def test_knn():
    # Generování 2D dat (two moons)
    X_2D, y_2D = make_moons(n_samples=300, noise=0.1)
    knn = KNN(k=5)
    y_pred = knn.predict(X_2D, y_2D, X_2D)

    # Matplotlib vizualizace
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y_pred, cmap='viridis', s=10)
    plt.title("KNN - 2D")
    plt.show()

    # Plotly vizualizace
    fig = px.scatter(x=X_2D[:, 0], y=X_2D[:, 1], color=y_pred.astype(str))
    fig.show()

    # Generování 2D dat (dvě soustředné kružnice)
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5)
    y_pred_circles = knn.predict(X_circles, y_circles, X_circles)

    plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_pred_circles, cmap='viridis', s=10)
    plt.title("KNN - 2 soustředné kružnice")
    plt.show()

    fig = px.scatter(x=X_circles[:, 0], y=X_circles[:, 1], color=y_pred_circles.astype(str))
    fig.show()

    # Generování 3D dat
    X_3D, y_3D = make_blobs(n_samples=300, centers=3, n_features=3, random_state=42)
    y_pred_3d = knn.predict(X_3D, y_3D, X_3D)

    # Matplotlib 3D vizualizace
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2], c=y_pred_3d, cmap='viridis', s=10)
    plt.title("KNN - 3D")
    plt.show()

    # Plotly 3D vizualizace
    fig = px.scatter_3d(x=X_3D[:, 0], y=X_3D[:, 1], z=X_3D[:, 2], color=y_pred_3d.astype(str))
    fig.show()


def test_csv_knn(csv_file_path, k=5):
    """
    Zpracuje data z CSV souboru pomocí KNN algoritmu.
    Ignoruje poslední sloupec v CSV a používá předposlední jako cílové hodnoty.

    Args:
        csv_file_path (str): Cesta k CSV souboru
        k (int): Počet sousedů pro KNN algoritmus (výchozí: 5)
    """
    # Načtení dat z CSV
    print(f"Načítám data z: {csv_file_path}")
    data = pd.read_csv(csv_file_path)

    # Kontrola datových typů
    print("Datové typy sloupců:")
    print(data.dtypes)

    # Ignorujeme poslední dva sloupce (point_type a cluster_id) a použijeme pouze dimenze
    # Předpokládáme, že sloupce dimenzí jsou na začátku
    dimension_cols = [col for col in data.columns if col.startswith('dim_')]

    if not dimension_cols:
        # Pokud nebyly nalezeny sloupce s prefixem 'dim_', použijeme všechny sloupce kromě posledních dvou
        X = data.iloc[:, :-2].values
    else:
        # Jinak použijeme pouze nalezené sloupce dimenzí
        X = data[dimension_cols].values

    # Pro účely KNN potřebujeme také cílové hodnoty (y)
    # Vytvoříme jednoduché umělé cílové hodnoty - všechny body patří do jedné třídy
    y = np.zeros(len(X))

    print(f"Používám {X.shape[1]} dimenzí pro analýzu, ignoruji poslední dva sloupce.")
    print(f"Tvar matice X: {X.shape}")

    # Ujistíme se, že všechny hodnoty jsou numerické
    X = X.astype(float)

    print(f"Načteno {X.shape[0]} vzorků s {X.shape[1]} dimenzemi.")
    print(f"Poslední sloupec (terica) byl ignorován.")

    # Inicializace a použití KNN klasifikátoru
    knn = KNN(k=k)
    y_pred = knn.predict(X, y, X)

    # Zjištění dimenze dat pro vizualizaci
    dimensions = X.shape[1]

    if dimensions == 2:
        # 2D vizualizace pomocí Matplotlib
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=10)
        plt.title(f"KNN - CSV data (k={k})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Třída")
        plt.show()

        # 2D vizualizace pomocí Plotly
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y_pred.astype(str),
                         labels={"x": "Feature 1", "y": "Feature 2", "color": "Predikovaná třída"})
        fig.update_layout(title=f"KNN - CSV data (k={k})")
        fig.show()

    elif dimensions == 3:
        # 3D vizualizace pomocí Matplotlib
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, cmap='viridis', s=10)
        plt.title(f"KNN - CSV data 3D (k={k})")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        plt.colorbar(scatter, label="Třída")
        plt.show()

        # 3D vizualizace pomocí Plotly
        fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y_pred.astype(str),
                            labels={"x": "Feature 1", "y": "Feature 2", "z": "Feature 3", "color": "Predikovaná třída"})
        fig.update_layout(title=f"KNN - CSV data 3D (k={k})")
        fig.show()

    else:
        print(f"Data mají {dimensions} dimenzí, což není možné jednoduše vizualizovat.")
        print("Výsledky klasifikace:")
        unique_classes, counts = np.unique(y_pred, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            print(f"Třída {cls}: {count} prvků")

# Hlavní funkce pro spuštění testů
if __name__ == "__main__":
    # Odkomentujte následující kód pro zpracování CSV souboru
    test_csv_knn('data/shapes/shapes_dataset.csv', k=3)  # Můžete změnit hodnotu k podle potřeby

    # Nebo použijte původní testy na syntetických datech
    # test_knn()