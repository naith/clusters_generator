import numpy as np
import random
import plotly.graph_objects as go

def generate_poisson_disk_samples_exact(dimensions, num_samples, min_val, max_val, radius_factor=1.0, max_attempts=100):
    """
    Generuje vzorky Poissonova disku a snaží se dosáhnout přesného počtu.

    Args:
        dimensions: Počet dimenzí.
        num_samples: Požadovaný počet vzorků.
        min_val: Minimální hodnota pro každou dimenzi.
        max_val: Maximální hodnota pro každou dimenzi.
        radius_factor: Faktor pro úpravu poloměru.
        max_attempts: Maximální počet pokusů o vygenerování vzorků.

    Returns:
        Numpy array s vygenerovanými vzorky.
        Vrací None, pokud se nepodaří vygenerovat alespoň minimum vzorků.
    """
    if min_val >= max_val:
        raise ValueError("min_val musí být menší než max_val")
    if num_samples <= 0 or dimensions <= 0:
        raise ValueError("num_samples a dimensions musí být kladné")

    samples = []
    for _ in range(max_attempts): # Opakujeme generování, pokud nedosáhneme cíle
        current_samples = generate_poisson_disk_samples(dimensions, num_samples*2, min_val, max_val, radius_factor) # Zkusíme vygenerovat více bodů
        if current_samples is not None:
            if len(current_samples) >= num_samples:
                samples = current_samples[:num_samples] # Vybereme prvních num_samples bodů
                return np.array(samples)
    if not samples:
        print(f"Nepodařilo se vygenerovat ani minimum {num_samples} bodů ani po {max_attempts} pokusech.")
        return None

    return np.array(samples)


def generate_poisson_disk_samples(dimensions, num_samples, min_val, max_val, radius_factor=1.0):
    """
    Generuje náhodné multidimenzionální vzorky pomocí Poissonova disku (Blue noise).

    Args:
        dimensions: Počet dimenzí.
        num_samples: Cílový počet vzorků (nemusí být dosažen).
        min_val: Minimální hodnota pro každou dimenzi.
        max_val: Maximální hodnota pro každou dimenzi.
        radius_factor: Faktor pro úpravu poloměru (ovlivňuje hustotu vzorků).

    Returns:
        Numpy array s vygenerovanými vzorky, nebo None v případě chyby.
    """

    if min_val >= max_val:
        raise ValueError("min_val musí být menší než max_val")
    if num_samples <= 0 or dimensions <= 0:
        raise ValueError("num_samples a dimensions musí být kladné")

    # Velikost prostoru a mřížky
    space_size = np.array([max_val - min_val] * dimensions)
    radius = (np.prod(space_size) / num_samples) ** (1.0 / dimensions) * radius_factor  # Odhad poloměru
    cell_size = radius / np.sqrt(dimensions)
    grid_size = np.ceil(space_size / cell_size).astype(int)

    grid = np.full(tuple(grid_size), -1, dtype=int)
    samples = []
    active_list = []

    # Vygenerování prvního vzorku
    first_sample = np.random.uniform(min_val, max_val, dimensions)
    samples.append(first_sample)
    grid_index = (np.floor((first_sample - min_val) / cell_size)).astype(int)
    grid[tuple(grid_index)] = 0
    active_list.append(0)

    while active_list:
        index = random.choice(active_list)
        sample = samples[index]

        found = False
        for _ in range(30):  # Zkoušíme 30 náhodných bodů kolem
            new_sample = sample + np.random.uniform(-radius, radius, dimensions)

            # Omezení na prostor
            if np.all(new_sample >= min_val) and np.all(new_sample <= max_val):
                new_grid_index = (np.floor((new_sample - min_val) / cell_size)).astype(int)

                # Kontrola okolí v mřížce
                valid = True
                for offset in np.ndindex(*([3] * dimensions)):
                    neighbor_index = new_grid_index - 1 + np.array(offset)
                    if np.all(neighbor_index >= 0) and np.all(neighbor_index < grid_size):
                        if grid[tuple(neighbor_index)] != -1:
                            neighbor = samples[grid[tuple(neighbor_index)]]
                            if np.linalg.norm(new_sample - neighbor) < radius:
                                valid = False
                                break
                if valid:
                    samples.append(new_sample)
                    grid[tuple(new_grid_index)] = len(samples) - 1
                    active_list.append(len(samples) - 1)
                    found = True
                    break
        if not found:
            active_list.remove(index)

    return np.array(samples)


dimensions = 2 # Pro vizualizaci Plotly musí být 2D
num_samples = 6
min_val = -600
max_val = 600

try:
    samples = generate_poisson_disk_samples(dimensions, num_samples, min_val, max_val)
    if samples is not None:
        print(f"Vygenerováno {len(samples)} vzorků.")

        # Vizualizace pomocí Plotly
        fig = go.Figure(data=[go.Scatter(
            x=samples[:, 0],
            y=samples[:, 1],
            mode='markers',  # Zobrazí body jako značky
            marker=dict(
                size=5,       # Velikost značek
                color='blue', # Barva značek
                opacity=0.8   # Průhlednost značek
            )
        )])

        fig.update_layout(
            title="Poisson Disk Sampling (Blue Noise)",
            xaxis_title="X",
            yaxis_title="Y",
            xaxis=dict(scaleanchor="y", scaleratio=1), # Zajistí stejné měřítko os
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        fig.show() # Zobrazí interaktivní graf

    else:
        print("Generování selhalo.")
except ValueError as e:
    print(f"Chyba: {e}")

dimensions = 2
num_samples = 6 # Chceme přesně 6 bodů, pokud to bude možné
min_val = 0
max_val = 10

try:
    samples = generate_poisson_disk_samples_exact(dimensions, num_samples, min_val, max_val)
    if samples is not None:
        print(f"Vygenerováno {len(samples)} vzorků.")

        # Vizualizace pomocí Plotly (stejná jako v předchozí verzi)
        fig = go.Figure(data=[go.Scatter(
            x=samples[:, 0],
            y=samples[:, 1],
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.8)
        )])
        fig.update_layout(title="Poisson Disk Sampling (Blue Noise)", xaxis_title="X", yaxis_title="Y", xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x", scaleratio=1))
        fig.show()

    else:
        print("Generování selhalo.")

except ValueError as e:
    print(f"Chyba: {e}")