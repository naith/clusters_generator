import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Tuple, List, Optional


def generate_nd_rosenbrock(
        n_dimensions: int = 2,
        points_per_dim: int = 100,
        range_per_dim: Tuple[float, float] = (-10, 10),
        min_offset: float = -10.0  # Offset pro minimální hodnotu funkce
) -> pd.DataFrame:
    """
    Generuje N-dimenzionální Rosenbrock funkci (banánovou funkci).

    Parametry:
    ----------
    n_dimensions : int
        Počet dimenzí (minimálně 2)
    points_per_dim : int
        Počet bodů na jednu dimenzi
    range_per_dim : Tuple[float, float]
        Rozsah hodnot pro každou dimenzi
    min_offset : float
        Offset pro minimum funkce (posun celé funkce nahoru/dolů)
    """
    if n_dimensions < 2:
        raise ValueError("Počet dimenzí musí být alespoň 2")

    # Vytvoření bodů pro každou dimenzi
    coordinates = [np.linspace(range_per_dim[0], range_per_dim[1], points_per_dim)
                   for _ in range(n_dimensions)]

    # Vytvoření n-dimenzionální mřížky
    grid = np.meshgrid(*coordinates)

    # Výpočet Rosenbrock funkce
    values = np.zeros_like(grid[0])

    for i in range(n_dimensions - 1):
        x_i = grid[i]
        x_i_plus_1 = grid[i + 1]
        values += 100.0 * (x_i_plus_1 - x_i ** 2) ** 2 + (1 - x_i) ** 2

    # Škálování hodnot do rozsahu [min_offset, 10]
    min_val = np.min(values)
    max_val = np.max(values)
    scale_range = 10.0 - min_offset
    values_scaled = (values - min_val) / (max_val - min_val) * scale_range + min_offset

    # Vytvoření DataFrame
    data_dict = {}
    for i, dim_coords in enumerate(grid):
        data_dict[f'dim_{i + 1}'] = dim_coords.flatten()
    data_dict['value'] = values_scaled.flatten()

    return pd.DataFrame(data_dict)


def visualize_rosenbrock_data(
        df: pd.DataFrame,
        min_offset: float = -10.0,  # Přidán parametr min_offset pro titulek
        range_per_dim: Tuple[float, float] = (-10, 10)
) -> Optional[go.Figure]:
    """
    Vizualizuje Rosenbrock funkci pro 2D a 3D případy.
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
                    showscale=True,
                    cmin=-10,
                    cmax=10
                )
            )
        ])

    fig.update_layout(
        title=f'{n_dimensions}D Rosenbrock funkce (min offset: {min_offset})',
        scene=dict(
            xaxis_title='Dimenze 1',
            yaxis_title='Dimenze 2',
            zaxis_title='Dimenze 3' if n_dimensions == 3 else 'Hodnota',
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            zaxis=dict(range=[-10, 10]),
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
    # Test generování dat pro různé dimenze a offsety
    dimensions = [2, 3, 4]
    min_offsets = [-10, -5, 0]  # Různé offsety pro testování

    for n_dim in dimensions:
        for min_offset in min_offsets:
            print(f"\nGenerování {n_dim}D Rosenbrock funkce s offsetem {min_offset}...")

            # Generování dat
            df = generate_nd_rosenbrock(
                n_dimensions=n_dim,
                points_per_dim=20 if n_dim > 2 else 50,
                min_offset=min_offset
            )

            # Export dat
            filename = f'rosenbrock_{n_dim}d_offset_{int(min_offset)}.csv'
            df.to_csv(filename, index=False)
            print(f"Data uložena do: {filename}")
            print(f"Tvar datasetu: {df.shape}")

            # Vizualizace
            fig = visualize_rosenbrock_data(df, min_offset=min_offset)
            if fig is not None:
                fig.show()
                print(f"Vizualizace {n_dim}D dat vytvořena")

            # Základní statistiky
            print("\nZákladní statistiky:")
            print(df.describe())