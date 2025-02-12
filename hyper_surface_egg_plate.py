import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Tuple, List, Optional


def generate_nd_eggplate(
        n_dimensions: int = 2,
        n_waves: int = 3,
        min_depth: float = 5.0,
        flat_threshold: float = 0.8,  # Práh pro zploštění (0-1)
        points_per_dim: int = 100,
        range_per_dim: Tuple[float, float] = (-10, 10)
) -> pd.DataFrame:
    """
    Generuje N-dimenzionální zvlněný povrch s plochými maximy.

    Parametry:
    ----------
    n_dimensions : int
        Počet dimenzí (minimálně 2)
    n_waves : int
        Počet vln na každou dimenzi
    min_depth : float
        Hloubka minim
    flat_threshold : float
        Práh pro zploštění maxim (0-1), vyšší hodnota = více ploché
    points_per_dim : int
        Počet bodů na jednu dimenzi
    range_per_dim : Tuple[float, float]
        Rozsah hodnot pro každou dimenzi
    """
    if n_dimensions < 2:
        raise ValueError("Počet dimenzí musí být alespoň 2")

    # Vytvoření bodů pro každou dimenzi
    coordinates = [np.linspace(range_per_dim[0], range_per_dim[1], points_per_dim)
                   for _ in range(n_dimensions)]

    # Vytvoření n-dimenzionální mřížky
    grid = np.meshgrid(*coordinates)

    # Výpočet zvlněného povrchu
    wave_frequency = n_waves * np.pi / (range_per_dim[1] - range_per_dim[0])
    values = np.zeros_like(grid[0])

    # Přidání sinusových vln pro každou dimenzi
    for dim_coords in grid:
        values += np.sin(wave_frequency * dim_coords)

    # Normalizace podle počtu dimenzí
    values = values / n_dimensions

    # Úprava tvaru - zploštění maxim
    values_normalized = values / np.max(np.abs(values))

    # Aplikujeme zploštění pouze na pozitivní hodnoty
    mask_positive = values_normalized > flat_threshold
    values_normalized[mask_positive] = flat_threshold

    # Škálování na požadovanou hloubku
    values = values_normalized * min_depth

    # Vytvoření DataFrame
    data_dict = {}
    for i, dim_coords in enumerate(grid):
        data_dict[f'dim_{i + 1}'] = dim_coords.flatten()
    data_dict['value'] = values.flatten()

    return pd.DataFrame(data_dict)


def visualize_surface_data(
        df: pd.DataFrame,
        n_waves: int,
        min_depth: float,
        range_per_dim: Tuple[float, float] = (-10, 10)
) -> Optional[go.Figure]:
    """
    Vizualizuje data pro 2D a 3D případy.
    """
    n_dimensions = len([col for col in df.columns if col.startswith('dim_')])
    if n_dimensions not in [2, 3]:
        print(f"Vizualizace není dostupná pro {n_dimensions}D data")
        return None

    if n_dimensions == 2:
        points_per_dim = int(np.sqrt(len(df)))
        grid_x = df['dim_1'].values.reshape(points_per_dim, points_per_dim)
        grid_y = df['dim_2'].values.reshape(points_per_dim, points_per_dim)
        grid_z = df['value'].values.reshape(points_per_dim, points_per_dim)

        fig = go.Figure(data=[
            go.Surface(
                x=grid_x,
                y=grid_y,
                z=grid_z,
                colorscale='Viridis',
                contours={
                    "z": {
                        "show": True,
                        "start": -10,  # Pevný rozsah pro vrstevnice
                        "end": 10,
                        "size": 2,
                        "width": 2,
                        "color": "white",
                        "project_z": True
                    }
                }
            )
        ])
    else:  # 3D
        fig = go.Figure(data=[
            go.Scatter3d(
                x=df['dim_1'],
                y=df['dim_2'],
                z=df['dim_3'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df['value'],
                    colorscale='Viridis',
                    showscale=True,
                    cmin=-10,  # Pevný rozsah pro barevnou škálu
                    cmax=10
                )
            )
        ])

    # Nastavení vzhledu - všechny osy pevně -10,10
    fig.update_layout(
        title=f'{n_dimensions}D eggplate s plochými maximy (vlny: {n_waves}, hloubka: {min_depth})',
        scene=dict(
            xaxis_title='Dimenze 1',
            yaxis_title='Dimenze 2',
            zaxis_title='Dimenze 3' if n_dimensions == 3 else 'Hodnota',
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            zaxis=dict(range=[-10, 10]),  # Pevný rozsah
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        width=800,
        height=800,
    )

    return fig


if __name__ == "__main__":
    # Test generování dat pro různé dimenze a parametry
    dimensions = [2, 3]
    n_waves_values = [4, 10]
    min_depth = 1.0
    flat_threshold = 0.5

    for n_dim in dimensions:
        for n_waves in n_waves_values:
            print(f"\nGenerování {n_dim}D dat s {n_waves} vlnami...")

            # Generování dat
            df = generate_nd_eggplate(
                n_dimensions=n_dim,
                n_waves=n_waves,
                min_depth=min_depth,
                flat_threshold=flat_threshold,
                points_per_dim=20 if n_dim > 2 else 50
            )

            # Export dat
            filename = f'eggplate_flat_{n_dim}d_waves_{n_waves}.csv'
            df.to_csv(f"data/hyper_surface/{filename}", index=False)
            print(f"Data uložena do: {filename}")
            print(f"Tvar datasetu: {df.shape}")

            # Vizualizace
            fig = visualize_surface_data(df, n_waves, min_depth)
            if fig is not None:
                fig.show()
                print(f"Vizualizace {n_dim}D dat vytvořena")

            # Základní statistiky
            print("\nZákladní statistiky:")
            print(df.describe())