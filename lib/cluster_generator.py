import numpy as np
import plotly.graph_objects as go


class ClusterGenerator:

    def __init__(self, waypoints, tension, cluster_density):

        self.clusters = None
        self.spread = None
        self.centers = None
        self.waypoints = np.array(waypoints)
        self.tension = tension
        self.cluster_density = cluster_density

    def compute_constant_spread(self, centers, diameter_factor=0.8):

        dist = np.linalg.norm(centers[1] - centers[0])
        diameter = dist * diameter_factor
        radius = diameter / 2.0
        return radius

    def create_cluster_3d(self, center, spread=0.01, distribution='gauss'):

        center = np.array(center)

        if distribution == 'gauss':
            # Pro gaussovské rozložení používáme spread jako poloměr
            # a spread/3 jako směrodatnou odchylku
            points = np.random.normal(0, spread / 3, size=(self.cluster_density, 3))
        else:
            # Pro uniformní rozložení generujeme body rovnoměrně v kouli
            points = []
            while len(points) < self.cluster_density:
                point = np.random.uniform(-spread, spread, 3)
                if np.linalg.norm(point) <= spread:
                    points.append(point)
            points = np.array(points)

        return center + points

    def generate_shape(self, tension, distribution, radius=1, distance_ratio=0.125):

        self.spread = radius

        if len(self.waypoints) == 1:
            # Pro jediný bod - vytváříme kulový cluster
            self.centers = self.waypoints
            self.clusters = [
                self.create_cluster_3d(
                    self.centers[0],
                    spread=self.spread,
                    distribution=distribution
                )
            ]
            self.clusters = np.vstack(self.clusters)
            return

        # Pro více bodů - generujeme body s upravenou vzdáleností
        center_distance = radius * distance_ratio  # Vzdálenost mezi centry
        self.centers = self.const_distance_uniform_catmull_rom_spline(
            radius=center_distance,  # Používáme upravenou vzdálenost pro centra
            tension=tension
        )

        # Generování clusterů - zde stále používáme původní radius
        self.clusters = []
        for c in self.centers:
            self.clusters.append(
                self.create_cluster_3d(c, spread=self.spread,
                                       distribution=distribution)
            )
        self.clusters = np.vstack(self.clusters)

    def add_scatter_traces(self, fig, color, name_prefix,
                           show_waypoints=True, show_lines=True,
                           show_centers=True, show_clusters=True):

        if self.waypoints.ndim > 3:
            return

        if show_waypoints:
            fig.add_trace(go.Scatter3d(
                x=self.waypoints[:, 0],
                y=self.waypoints[:, 1],
                z=self.waypoints[:, 2],
                mode='markers',
                marker=dict(size=4, color=color),
                name=f'{name_prefix} Waypoints'
            ))

        if show_centers:
            fig.add_trace(go.Scatter3d(
                x=self.waypoints[:, 0],
                y=self.waypoints[:, 1],
                z=self.waypoints[:, 2],
                mode='markers',
                marker=dict(size=4, color=color),
                name=f'{name_prefix} Waypoints'
            ))

        if show_lines:
            fig.add_trace(go.Scatter3d(
                x=self.centers[:, 0],
                y=self.centers[:, 1],
                z=self.centers[:, 2],
                mode='lines+markers',
                marker=dict(size=2, color=color),
                line=dict(width=2, color=color),
                name=f'{name_prefix} Curve (centers)'
            ))

        if show_clusters:
            fig.add_trace(go.Scatter3d(
                x=self.clusters[:, 0],
                y=self.clusters[:, 1],
                z=self.clusters[:, 2],
                mode='markers',
                marker=dict(size=2, color=color, opacity=1),
                name=f'{name_prefix} Clusters'
            ))

    def transform_nd_axes(self, angles, scales, translation):

        if self.waypoints.ndim != 2:
            raise ValueError("Param 'Points' must be a 2D nut (n, n).")
        n = self.waypoints.shape[1]

        if len(angles) != n:
            raise ValueError(f"Number of angles ({len(angles)}) != n={n}.")
        if len(scales) != n:
            raise ValueError(f"Number of scaling factors ({len(scales)}) != n={n}.")
        if len(translation) != n:
            raise ValueError(f"The number of translation components ({len(translation)}) != n={n}.")

        if n < 3:
            raise ValueError(
                "It makes no sense for n <3.")

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

    def transform_clusters(self, rotations, scales, translation):

        if self.clusters is None or self.centers is None:
            raise ValueError("You must generate cluster using Generate_Shape first")

        n = self.clusters.shape[1]

        # Checking input parameters
        if len(rotations) != n or len(scales) != n or len(translation) != n:
            raise ValueError(f"All transformation parameters must have a length {n}")

        # We will use the well -known cluster center
        center = self.centers[0]  # Pro jeden cluster máme jedno centrum

        # We move cluster to the beginning of coordinates with respect to the known center
        points = self.clusters - center

        # We apply scaling
        S = np.diag(scales)
        points = points @ S

        # We apply rotations
        for axis in range(n):
            if rotations[axis] == 0:
                continue

            angle = np.radians(rotations[axis])
            plane_axes = [i for i in range(n) if i != axis][:2]

            if len(plane_axes) < 2:
                continue

            R = np.eye(n)
            i, j = plane_axes
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            R[i, i] = cos_a
            R[i, j] = -sin_a
            R[j, i] = sin_a
            R[j, j] = cos_a

            points = points @ R

        # We will return the cluster to the original position
        points = points + center

        # Apply an additional translation
        points = points + translation

        self.clusters = points

    def const_distance_uniform_catmull_rom_spline(self, radius, tension=0.165):

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

    def bezier_point(self, p0, p1, p2, p3, t):

        return ((1 - t) ** 3 * p0 +
                3 * (1 - t) ** 2 * t * p1 +
                3 * (1 - t) * t ** 2 * p2 +
                t ** 3 * p3)

    def catmull_rom_to_bezier(self, p0, p1, p2, p3, tension=0.5):

        factor = (1 - tension) / 6.0
        d1 = (p2 - p0) * tension
        d2 = (p3 - p1) * tension
        C1 = p1 + d1
        C2 = p2 - d2
        return C1, C2
