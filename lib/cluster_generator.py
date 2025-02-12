import numpy as np
import plotly.graph_objects as go


class ClusterGenerator:
    """
    Třída pro generování clusterů bodů podél Catmull-Rom křivky.
    Umožňuje vytvářet shluky bodů s gaussovským nebo uniformním rozložením.
    """

    def __init__(self, waypoints, tension, cluster_size, diameter_factor):
        """
        Inicializace generátoru clusterů.

        Args:
            waypoints: Řídící body křivky
            tension: Napětí křivky (ovlivňuje její hladkost)
            cluster_size: Počet bodů v každém clusteru
            diameter_factor: Faktor pro výpočet průměru clusteru
        """
        self.clusters = None
        self.spread = None
        self.centers = None
        self.waypoints = np.array(waypoints)
        self.tension = tension
        self.cluster_size = cluster_size
        self.diameter_factor = diameter_factor

    def compute_constant_spread(self, centers, diameter_factor=0.8):
        """
        Vypočítá rozptyl pro všechny clustery na základě vzdálenosti mezi prvními dvěma body.

        Args:
            centers: Pole center clusterů
            diameter_factor: Faktor pro výpočet průměru clusteru
        Returns:
            float: Poloměr pro generování clusterů
        """
        dist = np.linalg.norm(centers[1] - centers[0])
        diameter = dist * diameter_factor
        radius = diameter / 2.0
        return radius

    def create_cluster_3d(self, center, spread=0.01, distribution='gauss'):
        """
        Vytvoří cluster bodů kolem daného centra.

        Args:
            center: Centrum clusteru
            spread: Rozptyl bodů od centra
            distribution: Typ rozložení ('gauss' nebo 'uniform')
        Returns:
            numpy.ndarray: Pole bodů clusteru
        """
        center = np.array(center)
        if distribution == 'gauss':
            offsets = np.random.normal(0, spread, size=(self.cluster_size, 3))
        else:
            offsets = np.random.uniform(-spread, spread, size=(self.cluster_size, 3))
        return center + offsets

    def generate_shape(self, tension, distribution, total_samples):
        """
        Generuje kompletní tvar složený z clusterů podél křivky.

        Nejprve vypočítá přibližnou křivku pro určení základního rozptylu,
        pak použije tento rozptyl pro přesné rozmístění bodů.

        Args:
            tension: Napětí křivky
            distribution: Typ rozložení bodů v clusterech
            total_samples: Celkový počet vzorků
        """
        # Nejprve vytvoříme hrubou aproximaci křivky pro výpočet rozptylu
        rough_centers = self.catmull_rom_spline_3d_uniform(
            self.waypoints,
            total_samples=max(10, len(self.waypoints) * 2),
            tension=tension
        )

        # Vypočítáme počáteční rozptyl z hrubé aproximace
        if len(rough_centers) >= 2:
            initial_dist = np.linalg.norm(rough_centers[1] - rough_centers[0])
            self.spread = initial_dist * self.diameter_factor / 2.0
        else:
            # Fallback pro případ příliš krátkých křivek
            self.spread = np.linalg.norm(self.waypoints[1] - self.waypoints[0]) * self.diameter_factor / 2.0

        # Nyní můžeme generovat body s přesnou vzdáleností
        self.centers = self.const_distance_uniform_catmull_rom_spline(
            radius=self.spread,
            tension=tension
        )

        # Generujeme clustery
        self.clusters = []
        for c in self.centers:
            self.clusters.append(
                self.create_cluster_3d(c, spread=self.spread,
                                       distribution=distribution)
            )
        self.clusters = np.vstack(self.clusters)

    def add_scatter_traces(self, fig, color, name_prefix):
        """
        Přidá vizualizační stopy do Plotly grafu.

        Args:
            fig: Plotly Figure objekt
            color: Barva pro vizualizaci
            name_prefix: Prefix pro názvy stop
        """
        if self.waypoints.ndim > 3:
            return
        fig.add_trace(go.Scatter3d(
            x=self.waypoints[:, 0],
            y=self.waypoints[:, 1],
            z=self.waypoints[:, 2],
            mode='markers',
            marker=dict(size=4, color=color),
            name=f'{name_prefix} Waypoints'
        ))
        fig.add_trace(go.Scatter3d(
            x=self.centers[:, 0],
            y=self.centers[:, 1],
            z=self.centers[:, 2],
            mode='lines+markers',
            marker=dict(size=2, color=color),
            line=dict(width=2, color=color),
            name=f'{name_prefix} Křivka (centra)'
        ))
        fig.add_trace(go.Scatter3d(
            x=self.clusters[:, 0],
            y=self.clusters[:, 1],
            z=self.clusters[:, 2],
            mode='markers',
            marker=dict(size=2, color=color, opacity=0.5),
            name=f'{name_prefix} Clustery'
        ))

    def transform_nd_axes(self, angles, scales, translation):
        """
        Transformuje body v n-dimenzionálním prostoru.

        Args:
            angles: Úhly rotace pro každou osu
            scales: Škálovací faktory pro každou osu
            translation: Vektor posunutí
        """
        if self.waypoints.ndim != 2:
            raise ValueError("Param 'points' musí být 2D matice (N, n).")
        n = self.waypoints.shape[1]

        if len(angles) != n:
            raise ValueError(f"Počet úhlů ({len(angles)}) != n={n}.")
        if len(scales) != n:
            raise ValueError(f"Počet škálovacích faktorů ({len(scales)}) != n={n}.")
        if len(translation) != n:
            raise ValueError(f"Počet složek translace ({len(translation)}) != n={n}.")

        if n < 3:
            raise ValueError(
                "Pro n < 3 nedává smysl 'osa-based' rotace.")

        new_points = np.copy(self.waypoints)

        def pick_plane(k, n):
            idx = [x for x in range(n) if x != k]
            return idx[0], idx[1]

        S = np.diag(scales)
        new_points = new_points @ S.T

        for k, alpha_deg in enumerate(angles):
            if alpha_deg == 0:
                continue
            alpha = np.radians(alpha_deg)

            i, j = pick_plane(k, n)
            G = np.eye(n)
            c = np.cos(alpha)
            s = np.sin(alpha)
            G[i, i] = c
            G[i, j] = -s
            G[j, i] = s
            G[j, j] = c

            new_points = new_points @ G.T

        new_points += np.array(translation)
        self.waypoints = new_points
        return

    def const_distance_uniform_catmull_rom_spline(self, radius, tension=0.165):
        """
        Generuje body na křivce s přesně konstantní vzdáleností R (poloměr clusteru).

        Args:
            radius: Požadovaná vzdálenost mezi body (poloměr clusteru)
            tension: Napětí křivky
        Returns:
            numpy.ndarray: Pole bodů na křivce s konstantní vzdáleností
        """
        wp = np.array(self.waypoints)
        n = len(wp)
        if n < 2:
            return wp

        result = [wp[0]]  # Začínáme prvním bodem

        def point_at_t(s, t):
            """Vypočítá bod na křivce pro dané s a t."""
            p0 = wp[max(0, s - 1)]
            p1 = wp[s]
            p2 = wp[s + 1]
            p3 = wp[min(n - 1, s + 2)]
            c1, c2 = self.catmull_rom_to_bezier(p0, p1, p2, p3, tension)
            return self.bezier_point(p1, c1, c2, p2, t)

        def find_next_point(last_point, s, t_start):
            """
            Najde další bod na křivce ve vzdálenosti radius od posledního bodu.
            Používá binární vyhledávání pro nalezení správného parametru t.
            """
            t_left = t_start
            t_right = 1.0
            iterations = 0
            best_t = t_start
            best_diff = float('inf')

            while iterations < 20:  # Maximální počet iterací
                t_mid = (t_left + t_right) / 2
                point = point_at_t(s, t_mid)
                dist = np.linalg.norm(point - last_point)
                diff = abs(dist - radius)

                if diff < best_diff:
                    best_diff = diff
                    best_t = t_mid

                if abs(dist - radius) < radius * 0.001:  # Tolerance 0.1%
                    return point, t_mid, True

                if dist < radius:
                    t_left = t_mid
                else:
                    t_right = t_mid

                iterations += 1

            return point_at_t(s, best_t), best_t, False

        # Generování bodů
        current_s = 0
        current_t = 0
        while current_s < n - 1:
            last_point = result[-1]
            next_point, new_t, found = find_next_point(last_point, current_s, current_t)

            if found:
                result.append(next_point)
                if new_t >= 1.0:
                    current_s += 1
                    current_t = 0
                else:
                    current_t = new_t
            else:
                current_s += 1
                current_t = 0
                if current_s >= n - 1:
                    break

        # Přidáme poslední bod, pokud jsme příliš daleko
        if np.linalg.norm(result[-1] - wp[-1]) > radius:
            result.append(wp[-1])

        return np.array(result)

    def adaptive_catmull_rom_spline(self, min_distance, tension=0.05):
        """
        Generuje body na Catmull-Rom křivce s adaptivním dělením.

        Args:
            min_distance: Minimální vzdálenost mezi body
            tension: Napětí křivky
        Returns:
            numpy.ndarray: Pole bodů na křivce
        """
        wp = np.array(self.waypoints)
        n = len(wp)
        if n < 2:
            return wp

        result = []

        def subdivide(p1, p2, s, tau):
            dist = np.linalg.norm(p2 - p1)
            if dist > min_distance:
                mid_tau = tau / 2
                t_global = s + mid_tau
                new_s = int(np.floor(t_global))
                new_tau = t_global - new_s
                p0 = wp[s - 1] if s - 1 >= 0 else p1
                p3 = wp[s + 2] if (s + 2) < n else p2
                C1, C2 = self.catmull_rom_to_bezier(p0, p1, p2, p3,
                                                    tension=tension)
                B = self.bezier_point(p1, C1, C2, p2, new_tau)

                subdivide(p1, B, s, new_tau)
                subdivide(B, p2, new_s, tau - new_tau)
            else:
                result.append(p1)

        for s in range((n - 1)):
            p1 = wp[s]
            p2 = wp[s + 1]
            subdivide(p1, p2, s, 1)

        result.append(wp[-1])
        return np.array(result)

    def bezier_point(self, p0, p1, p2, p3, t):
        """
        Vypočítá bod na kubické Bézierově křivce.

        Args:
            p0, p1, p2, p3: Kontrolní body křivky
            t: Parametr křivky (0 až 1)
        Returns:
            numpy.ndarray: Bod na křivce
        """
        return ((1 - t) ** 3 * p0 +
                3 * (1 - t) ** 2 * t * p1 +
                3 * (1 - t) * t ** 2 * p2 +
                t ** 3 * p3)

    def catmull_rom_spline_3d_uniform(self, waypoints, total_samples=100, tension=0.5):
        """
        Generuje uniformně vzorkované body na 3D Catmull-Rom křivce.

        Args:
            waypoints: Řídící body křivky
            total_samples: Celkový počet vzorků
            tension: Napětí křivky
        Returns:
            numpy.ndarray: Pole bodů na křivce
        """
        wp = np.array(waypoints)
        n = len(wp)
        if n < 2 or total_samples < 2:
            return wp

        result = []
        for i in range(total_samples):
            t_global = (n - 1) * i / (total_samples - 1)
            s = int(np.floor(t_global))
            tau = t_global - s

            if s >= n - 1:
                s = n - 2
                tau = 1.0

            p1 = wp[s]
            p2 = wp[s + 1]
            p0 = wp[s - 1] if s - 1 >= 0 else p1
            p3 = wp[s + 2] if (s + 2) < n else p2

            C1, C2 = self.catmull_rom_to_bezier(p0, p1, p2, p3,
                                                tension=tension)
            B = self.bezier_point(p1, C1, C2, p2, tau)
            result.append(B)

        return np.array(result)

    def catmull_rom_to_bezier(self, p0, p1, p2, p3, tension=0.5):
        """
        Převede kontrolní body Catmull-Rom křivky na kontrolní body Bézierovy křivky.

        Args:
            p0, p1, p2, p3: Kontrolní body Catmull-Rom křivky
            tension: Napětí křivky
        Returns:
            tuple: Dvojice kontrolních bodů pro Bézierovu křivku
        """
        factor = (1 - tension) / 6.0
        d1 = (p2 - p0) * tension
        d2 = (p3 - p1) * tension
        C1 = p1 + d1
        C2 = p2 - d2
        return C1, C2