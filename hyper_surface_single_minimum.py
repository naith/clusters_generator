import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Tuple, List, Optional


def generate_nd_surface_with_dip(
        n_dimensions: int = 2,
        crater_diameter: float = 10.0,
        min_depth: float = 10.0,
        points_per_dim: int = 100,
        range_per_dim: Tuple[float, float] = (-10, 10)
) -> pd.DataFrame:
    """
    Generuje N-dimenzionální data s důlkem uprostřed.

    Parametry:
    ----------
    n_dimensions : int
        Počet dimenzí (minimálně 2)
    crater_diameter : float
        Průměr ústí důlku (v jednotkách os)
    min_depth : float
        Hloubka důlku (absolutní hodnota minima)
    points_per_dim : int
        Počet bodů na jednu dimenzi
    range_per_dim : Tuple[float, float]
        Rozsah hodnot pro každou dimenzi

    Returns:
    --------
    pd.DataFrame
        DataFrame s vygenerovanými daty
    """
    if n_dimensions < 2:
        raise ValueError("Počet dimenzí musí být alespoň 2")

    # Vytvoření bodů pro každou dimenzi
    coordinates = [np.linspace(range_per_dim[0], range_per_dim[1], points_per_dim)
                   for _ in range(n_dimensions)]

    # Vytvoření n-dimenzionální mřížky
    grid = np.meshgrid(*coordinates)

    # Výpočet vzdálenosti od středu v n-dimenzionálním prostoru
    squared_distances = np.zeros_like(grid[0])
    for dim_coords in grid:
        squared_distances += dim_coords ** 2

    # Výpočet hodnot důlku
    sigma = crater_diameter / 4
    dip_values = -min_depth * np.exp(-squared_distances / (2 * sigma ** 2))

    # Vytvoření DataFrame
    data_dict = {}
    for i, dim_coords in enumerate(grid):
        data_dict[f'dim_{i + 1}'] = dim_coords.flatten()
    data_dict['value'] = dip_values.flatten()

    return pd.DataFrame(data_dict)


def visualize_surface_data(
        df: pd.DataFrame,
        crater_diameter: float,
        min_depth: float,
        range_per_dim: Tuple[float, float] = (-10, 10)
) -> Optional[go.Figure]:
    """
    Vizualizuje data pro 2D a 3D případy.

    Parametry:
    ----------
    df : pd.DataFrame
        DataFrame s vygenerovanými daty
    crater_diameter : float
        Průměr důlku (pro titulek)
    min_depth : float
        Hloubka důlku (pro titulek)
    range_per_dim : Tuple[float, float]
        Rozsah hodnot pro osy

    Returns:
    --------
    Optional[go.Figure]
        Plotly graf pro 2D/3D data nebo None pro vyšší dimenze
    """
    n_dimensions = len([col for col in df.columns if col.startswith('dim_')])
    if n_dimensions not in [2, 3]:
        print(f"Vizualizace není dostupná pro {n_dimensions}D data")
        return None

    # Příprava dat pro vizualizaci
    if n_dimensions == 2:
        # Reshape pro 2D surface plot
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
                        "start": -10,
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
                    showscale=True
                )
            )
        ])

    # Nastavení vzhledu
    axis_settings = dict(range=[range_per_dim[0], range_per_dim[1]])

    fig.update_layout(
        title=f'{n_dimensions}D data s důlkem (průměr: {crater_diameter}, hloubka: {min_depth})',
        scene=dict(
            xaxis_title='Dimenze 1',
            yaxis_title='Dimenze 2',
            zaxis_title='Dimenze 3' if n_dimensions == 3 else 'Hodnota',
            xaxis=axis_settings,
            yaxis=axis_settings,
            zaxis=axis_settings,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        width=1000,
        height=800,
    )

    return fig


if __name__ == "__main__":
    # Test generování dat pro různé dimenze
    dimensions = [2, 3, 4]
    crater_diameter = 8.0
    min_depth = 5.0

    for n_dim in dimensions:
        print(f"\nGenerování {n_dim}D dat...")

        # Generování dat
        df = generate_nd_surface_with_dip(
            n_dimensions=n_dim,
            crater_diameter=crater_diameter,
            min_depth=min_depth,
            points_per_dim=20 if n_dim > 2 else 50  # Méně bodů pro vyšší dimenze
        )

        # Export dat
        filename = f'surface_{n_dim}d_dip.csv'
        df.to_csv(f"data/hyper_surface/{filename}", index=False)
        print(f"Data uložena do: {filename}")
        print(f"Tvar datasetu: {df.shape}")

        # Vizualizace
        fig = visualize_surface_data(df, crater_diameter, min_depth)
        if fig is not None:
            fig.show()
            print(f"Vizualizace {n_dim}D dat vytvořena")

        # Základní statistiky
        print("\nZákladní statistiky:")
        print(df.describe())