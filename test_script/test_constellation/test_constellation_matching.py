#!/usr/bin/env python3
"""
Simulation of Starlink satellites with Earth rotation and Vispy visualization.

This script loads Starlink TLE data, filters invalid satellites, computes Earth’s rotation,
and visualizes both the Earth (with a textured sphere) and satellites in a 3D scene.
"""

import math
import time
import cProfile
import pstats
import logging
from itertools import combinations

import psutil
import numpy as np
from numba import njit
from PIL import Image
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import scipy
import networkx as nx

from skyfield.api import load
from skyfield.sgp4lib import TEME
from sgp4.api import SatrecArray
from vispy import app, scene
from vispy.geometry import MeshData, create_sphere
from vispy.visuals.filters import TextureFilter
from vispy.visuals.transforms import MatrixTransform, STTransform
from vispy import gloo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@njit
def greedy_max_weight_matching(E: np.ndarray) -> list:
    """
    Compute a greedy heuristic maximum weight matching for a NumPy array of edges.
    
    Parameters:
        E (np.ndarray): Array with shape (num_edges, 3) where each row is [u, v, weight].
    
    Returns:
        list: List of tuples (u, v) representing the selected edges.
    """
    sorted_indices = np.argsort(-E[:, 2])
    E_sorted = E[sorted_indices]
    
    matching = []
    matched_nodes = set()
    
    for edge in E_sorted:
        u, v, weight = edge
        u, v = int(u), int(v)
        if u not in matched_nodes and v not in matched_nodes:
            matching.append((u, v))
            matched_nodes.add(u)
            matched_nodes.add(v)
    
    return matching


def compute_face_texcoords(vertices: np.ndarray) -> list:
    """
    Compute texture coordinates for a single face of vertices.

    Parameters:
        vertices (np.ndarray): Array of vertex coordinates for one face.

    Returns:
        list: List of [u, v] texture coordinates for the face.
    """
    has_negative = any(
        p[0] < 0 and q[0] < 0 and p[1] * q[1] < 0 for p, q in combinations(vertices, 2)
    )
    face_texcoords = []
    for v in vertices:
        x, y, z = v
        theta = np.arctan2(y, x)
        phi = np.arccos(z / np.linalg.norm(v))
        if has_negative and theta < 0:
            theta += 2 * np.pi
        u = (theta + np.pi) / (2 * np.pi) / 2
        v_coord = phi / np.pi
        face_texcoords.append([u, v_coord])
    return face_texcoords


def compute_texcoords(faces: np.ndarray) -> np.ndarray:
    """
    Compute texture coordinates for all faces.

    Parameters:
        faces (np.ndarray): Array where each element represents a face's vertices.

    Returns:
        np.ndarray: Array of texture coordinates.
    """
    texcoords = []
    for face in faces:
        texcoords.extend(compute_face_texcoords(face))
    return np.array(texcoords)


def load_starlink_data(url: str, reload: bool = True) -> tuple:
    """
    Load Starlink TLE data from the provided URL.

    Parameters:
        url (str): URL to the TLE file.
        reload (bool): Whether to reload the TLE data.

    Returns:
        tuple: (timescale, valid_satellites, sat_array)
    """
    ts = load.timescale()
    satellites = load.tle_file(url, reload=reload)
    if not satellites:
        raise Exception("No Starlink satellites were loaded; check the TLE URL.")
    logger.info("Loaded %d Starlink satellites from %s", len(satellites), url)

    valid_satellites = []
    for sat in satellites:
        pos = sat.at(ts.now())
        if np.isnan(pos.position.km).any():
            message = pos.message if pos.message else "position is invalid"
            logger.warning("Skipping %s due to error: %s", sat.name, message)
            continue
        valid_satellites.append(sat)

    models = [sat.model for sat in valid_satellites]
    sat_array = SatrecArray(models)
    return ts, valid_satellites, sat_array


def setup_visualization() -> dict:
    """
    Set up the Vispy visualization environment including canvas, view, sphere, and markers.

    Returns:
        dict: Dictionary containing references to visualization components.
    """
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor=(1.0, 1.0, 1.0, 0))
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, elevation=45, distance=2.5)

    # # Set up the OpenGL state for non-transparent objects.
    # gloo.set_state(depth_test=True, depth_mask=True, blend=False, cull_face=False)

    # Global arrow for reference
    global_arrow = scene.visuals.Arrow(
        pos=np.array([[0, 0, 0], [1, 1, 1]]), color='black', width=3, arrow_size=20,
        arrow_type='stealth', parent=view.scene
    )
    view.add(global_arrow)

    # Axes with scaling.
    axes = scene.visuals.XYZAxis(parent=view.scene)
    view.add(axes)
    axes.transform = STTransform(scale=(2, 2, 2))

    # Load texture image for the Earth sphere.
    texture_path = "land_sea_texture.png"
    try:
        texture_image = Image.open(texture_path)
    except Exception as e:
        logger.error("Error loading texture image: %s", e)
        raise

    texture = np.array(texture_image)
    # Duplicate texture horizontally.
    texture = np.hstack([texture, texture])
    # Create sphere mesh and compute texture coordinates.
    sphere = create_sphere(rows=20, cols=40, radius=1, method='latitude', offset=False)
    vertices = sphere.get_vertices(indexed='faces')
    texcoords = compute_texcoords(vertices)

    mesh_data = MeshData(vertices=vertices)
    texture_filter = TextureFilter(texture, texcoords)

    sphere_visual = scene.visuals.Mesh(
        meshdata=mesh_data, color=(1.0, 1.0, 1.0, 1.0), shading=None
    )
    sphere_visual.attach(texture_filter)
    sphere_visual.set_gl_state('opaque')
    view.add(sphere_visual)

    # Apply transformation to the sphere.
    sphere_visual.transform = MatrixTransform()

    # Enable depth testing for transparent objects.
    gloo.set_state(depth_test=True, depth_mask=True, blend=True,
               blend_func=('src_alpha', 'one_minus_src_alpha'))
    
    # Create markers for satellite positions.
    scatter = scene.visuals.Markers()
    view.add(scatter)

    # Satellite arrows for LT directions.
    satellite_arrow = scene.visuals.Arrow()
    view.add(satellite_arrow)

    # Lines for satellite LISL.
    p_lisl = scene.visuals.Arrow()
    view.add(p_lisl)

    # Lines for connected LISL.
    c_lisl = scene.visuals.Arrow()
    view.add(c_lisl)

    return {
        "canvas": canvas,
        "view": view,
        "sphere_visual": sphere_visual,
        "scatter": scatter,
        "satellite_arrow": satellite_arrow,
        "p_lisl": p_lisl,
        "c_lisl": c_lisl,
    }


def expand_edges_with_original(edges: np.ndarray, repeat_per_node: int = 4) -> tuple:
    """
    Expand edges by duplicating each edge in a grid fashion.

    Parameters:
        edges (np.ndarray): Array of edges.
        repeat_per_node (int): Number of repetitions per node.

    Returns:
        tuple: (repeated_original, expanded_edges)
    """
    grid = np.stack(np.meshgrid(np.arange(repeat_per_node), np.arange(repeat_per_node),
                                indexing='ij'), axis=-1).reshape(-1, 2)
    expanded_edges = (edges[:, None, :] * repeat_per_node + grid[None, :, :]).reshape(-1, 2)
    repeated_original = np.repeat(edges, repeat_per_node * repeat_per_node, axis=0)
    return repeated_original, expanded_edges


class Simulation:
    FOR_THETA: float = 15.0  # Angle in degrees for the satellite LT direction.
    LISL_MAX_DISTANCE: float = 3000.0  # Maximum distance for LISL in km.
    TIME_SCALE: float = 10.0
    EARTH_RADIUS: float = 6371.0  # Earth's radius in km.
    
    PLOT_POTENTIAL_LISL: bool = False    
    
    FRONT_COLOR = np.array([0, 0, 1, 1])
    BACK_COLOR = np.array([1, 0.75, 0, 1])
    RIGHT_COLOR = np.array([1, 0, 0, 1])
    LEFT_COLOR = np.array([0.05, 1, 0.05, 1])
    
    def __init__(self, ts, sat_array, sphere_visual, scatter, satellite_arrow, p_lisl, c_lisl, canvas):
        """
        Initialize the simulation.

        Parameters:
            ts: Skyfield timescale.
            sat_array: Vectorized satellite propagation array.
            sphere_visual: Vispy visual for the Earth sphere.
            scatter: Vispy visual for satellite markers.
            satellite_arrow: Vispy visual for satellite LT arrows.
            p_lisl: Vispy visual for primary LISL lines.
            c_lisl: Vispy visual for connected LISL lines.
            canvas: Vispy SceneCanvas.
        """
        self.ts = ts
        self.sat_array = sat_array
        self.sphere_visual = sphere_visual
        self.scatter = scatter
        self.arrow = satellite_arrow
        self.p_lisl = p_lisl
        self.c_lisl = c_lisl
        self.canvas = canvas

        self.simulation_start_time = self.ts.now()
        self.real_start_time = time.perf_counter()
        self.update_count = 0
        self.accumulated_update_time = 0

        # Initialize satellite positions and velocities.
        self.positions = None
        self.velocities = None
        # Initialize directional vectors.
        self.front = None
        self.back = None
        self.down = None
        self.right = None
        self.left = None

    def get_simulation_time(self):
        """
        Compute the current simulation time based on the time scaling factor.

        Returns:
            Skyfield Time: The current simulation time.
        """
        elapsed_real = time.perf_counter() - self.real_start_time
        elapsed_scaled = elapsed_real * self.TIME_SCALE
        delta_days = elapsed_scaled / 86400  # Convert seconds to days.
        new_tt_jd = self.simulation_start_time.tt + delta_days
        return self.ts.tt(jd=new_tt_jd)

    def compute_rotation(self) -> float:
        """
        Compute the initial rotation angle from the current GMST.

        Returns:
            float: Rotation angle in degrees.
        """
        t_now = self.get_simulation_time()
        gmst_hours = t_now.gmst
        rotation_angle_deg = gmst_hours * 15  # 15° per hour.
        logger.debug("GMST: %.2f hours, Rotation angle: %.2f degrees", gmst_hours, rotation_angle_deg)
        return rotation_angle_deg

    def _update_earth_rotation(self):
        """Update the Earth's rotation transformation."""
        self.sphere_visual.transform.reset()
        self.sphere_visual.transform.rotate(self.compute_rotation(), (0, 0, 1))

    def _update_satellite_positions(self):
        """Update satellite positions and velocities."""
        current_time = self.get_simulation_time()
        error_upd, pos_upd, vel_upd = self.sat_array.sgp4(
            np.array([current_time.whole]),
            np.array([current_time.ut1_fraction])
        )
        positions = np.array(pos_upd).reshape(-1, 3) / self.EARTH_RADIUS
        velocities = np.array(vel_upd).reshape(-1, 3) / self.EARTH_RADIUS

        R_icrs_to_teme = TEME.rotation_at(current_time)
        R_teme_to_icrs = R_icrs_to_teme.T
        self.positions = positions @ R_teme_to_icrs
        self.velocities = velocities @ R_teme_to_icrs

        self.scatter.set_data(self.positions, face_color=[0, 0, 0, 0.5], size=10, edge_width=0)

    def _update_satellite_arrows(self):
        """Update satellite LT direction arrows."""
        front = self.velocities / np.linalg.norm(self.velocities, axis=1)[:, None]
        back = -front
        down = self.positions / np.linalg.norm(self.positions, axis=1)[:, None]
        right = np.cross(down, front)
        left = -right

        self.front, self.back, self.down, self.right, self.left = front, back, down, right, left

        a_from = np.tile(self.positions, (4, 1))
        a_to = np.concatenate((front, back, right, left), axis=0) * 0.01 + a_from
        a_data = np.concatenate((a_from, a_to), axis=1).reshape(-1, 3)

        num_arrows = self.positions.shape[0] * 8
        arrow_color = np.zeros((num_arrows, 4))
        arrow_color[: num_arrows // 4, :] = self.FRONT_COLOR
        arrow_color[num_arrows // 4: num_arrows // 2, :] = self.BACK_COLOR
        arrow_color[num_arrows // 2: 3 * num_arrows // 4, :] = self.RIGHT_COLOR
        arrow_color[3 * num_arrows // 4:, :] = self.LEFT_COLOR

        self.arrow.set_data(pos=a_data, color=arrow_color, width=5, connect='segments')

    def _update_links(self):
        """Update satellite link visualizations using KDTree and matching."""
        tree_start = time.perf_counter()
        tree = cKDTree(self.positions)
        distance_threshold = self.LISL_MAX_DISTANCE / self.EARTH_RADIUS
        edges = tree.query_pairs(r=distance_threshold, output_type='ndarray')

        sat_p_i = self.positions[edges[:, 0]]
        sat_p_j = self.positions[edges[:, 1]]
        direction = sat_p_j - sat_p_i
        direction /= np.linalg.norm(direction, axis=1)[:, None]

        view_from_stack = np.concatenate((
            np.einsum('ij,ij->i', self.front[edges[:, 0]], direction)[:, None],
            np.einsum('ij,ij->i', self.back[edges[:, 0]], direction)[:, None],
            np.einsum('ij,ij->i', self.right[edges[:, 0]], direction)[:, None],
            np.einsum('ij,ij->i', self.left[edges[:, 0]], direction)[:, None]
        ), axis=1)
        i_j_indicator = view_from_stack > math.cos(math.radians(self.FOR_THETA))
        i_j_binary = i_j_indicator.sum(axis=1) > 0

        view_to_stack = np.concatenate((
            np.einsum('ij,ij->i', self.front[edges[:, 1]], -direction)[:, None],
            np.einsum('ij,ij->i', self.back[edges[:, 1]], -direction)[:, None],
            np.einsum('ij,ij->i', self.right[edges[:, 1]], -direction)[:, None],
            np.einsum('ij,ij->i', self.left[edges[:, 1]], -direction)[:, None]
        ), axis=1)
        j_i_indicator = view_to_stack > math.cos(math.radians(self.FOR_THETA))
        j_i_binary = j_i_indicator.sum(axis=1) > 0

        final_indicator = np.logical_and(i_j_binary, j_i_binary)
        possible_edges = edges[final_indicator]

        p_lisl_LT_pair = np.einsum('ij,ik->ijk', i_j_indicator[final_indicator],
                                    j_i_indicator[final_indicator]).reshape(-1)
        view_from_stack = view_from_stack[final_indicator]
        view_to_stack = view_to_stack[final_indicator]
        view_LT_pair = np.minimum(view_from_stack[:, :, None], view_to_stack[:, None, :]).reshape(-1)
        view_LT_pair = view_LT_pair[p_lisl_LT_pair].reshape(-1, 1)

        repeated_edges, expanded_edges = expand_edges_with_original(possible_edges)
        repeated_edges = repeated_edges[p_lisl_LT_pair]
        expanded_edges = expanded_edges[p_lisl_LT_pair]

        edges_color = np.concatenate((
            self.FRONT_COLOR[np.newaxis],
            self.BACK_COLOR[np.newaxis],
            self.RIGHT_COLOR[np.newaxis],
            self.LEFT_COLOR[np.newaxis]
        ), axis=0)
        expanded_from = expanded_edges[:, 0] % 4
        expanded_to = expanded_edges[:, 1] % 4
        edges_color_from = np.concatenate((edges_color[expanded_from], edges_color[expanded_from]), axis=1).reshape(-1, 4)
        edges_color_to = np.concatenate((edges_color[expanded_to], edges_color[expanded_to]), axis=1).reshape(-1, 4)
        edges_color_data = np.concatenate((edges_color_from, edges_color_to), axis=0)
        edges_color_data[:, 3] = 0.1

        if self.PLOT_POTENTIAL_LISL:
            p_from = self.positions[repeated_edges[:, 0]]
            p_to = self.positions[repeated_edges[:, 1]]
            p_mid = (p_from + p_to) / 2
            p_lisl_data_from = np.concatenate((p_from, p_mid), axis=1).reshape(-1, 3)
            p_lisl_data_to = np.concatenate((p_mid, p_to), axis=1).reshape(-1, 3)
            p_lisl_data = np.concatenate((p_lisl_data_from, p_lisl_data_to), axis=0)
            self.p_lisl.set_data(pos=p_lisl_data, color=edges_color_data, width=0.75, connect='segments')

        relative_speed = self.velocities[repeated_edges[:, 1]] - self.velocities[repeated_edges[:, 0]]
        relative_direction = self.positions[repeated_edges[:, 1]] - self.positions[repeated_edges[:, 0]]
        cross = np.cross(relative_speed, relative_direction)
        angular_speed = np.linalg.norm(cross, axis=1)[:, None] / np.linalg.norm(relative_direction, axis=1)[:, None]
        view_time_approx = (math.radians(self.FOR_THETA) - np.arccos(view_LT_pair)) / np.abs(angular_speed)
        included_edges = view_time_approx.reshape(-1) > 100
        weighted_edges = np.concatenate((expanded_edges[included_edges], view_time_approx[included_edges]), axis=1)
        matching = greedy_max_weight_matching(weighted_edges)

        m = len(matching)
        flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
                                  for x in pair), dtype=int, count=2*m)
        connected_edges = flat_array.reshape(-1, 2)
        connected_sat = connected_edges // 4

        edges_color_from = np.concatenate((edges_color[connected_edges[:, 0] % 4],
                                             edges_color[connected_edges[:, 0] % 4]), axis=1).reshape(-1, 4)
        edges_color_to = np.concatenate((edges_color[connected_edges[:, 1] % 4],
                                           edges_color[connected_edges[:, 1] % 4]), axis=1).reshape(-1, 4)
        edges_color_data = np.concatenate((edges_color_from, edges_color_to), axis=0)
        edges_color_data[:, 3] = 0.75

        p_from = self.positions[connected_sat[:, 0]] * 1.0001
        p_to = self.positions[connected_sat[:, 1]] * 1.0001
        p_mid = (p_from + p_to) / 2
        c_lisl_data_from = np.concatenate((p_from, p_mid), axis=1).reshape(-1, 3)
        c_lisl_data_to = np.concatenate((p_mid, p_to), axis=1).reshape(-1, 3)
        c_lisl_data = np.concatenate((c_lisl_data_from, c_lisl_data_to), axis=0)
        self.c_lisl.set_data(pos=c_lisl_data, color=edges_color_data, width=1.5, connect='segments')

        tree_end = time.perf_counter()
        logger.info("KDTree build time: %.6f seconds", tree_end - tree_start)
        logger.info("Number of edges: %d", edges.shape[0])
        logger.info("Number of possible edges: %d", possible_edges.shape[0])
        logger.info("Number of LISL edges: %d", expanded_edges.shape[0])

    def update(self, event):
        """
        Update function called on each timer tick to update the simulation.
        """
        start_time = time.perf_counter()
        self.sphere_visual.update()

        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.Process().memory_info().rss / 1e6
        logger.info(f"CPU Usage: {cpu_usage}%, Memory Usage: {mem_usage:.2f} MB")

        try:
            self._update_earth_rotation()
            self._update_satellite_positions()
            self._update_satellite_arrows()
            self._update_links()
        except Exception as e:
            logger.error("Unexpected error during update: %s", e)

        self.canvas.update()

        elapsed = time.perf_counter() - start_time
        self.update_count += 1
        self.accumulated_update_time += elapsed

        if self.update_count % 5 == 0:
            avg_time = self.accumulated_update_time / 60
            logger.info("Average update time over last 5 frames: %.6f seconds", avg_time)
            self.accumulated_update_time = 0


def main():
    # Load Starlink data.
    satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
    ts, valid_satellites, sat_array = load_starlink_data(satellite_url, reload=False)

    # Set up visualization.
    viz = setup_visualization()
    canvas = viz["canvas"]
    sphere_visual = viz["sphere_visual"]
    scatter = viz["scatter"]
    satellite_arrow = viz["satellite_arrow"]
    p_lisl = viz["p_lisl"]
    c_lisl = viz["c_lisl"]

    # Create simulation instance.
    simulation = Simulation(ts, sat_array, sphere_visual, scatter, satellite_arrow, p_lisl, c_lisl, canvas)

    # Set up a timer to update the simulation at roughly 60 FPS.
    timer = app.Timer(interval=1 / 60.0, connect=simulation.update, start=True)

    if __name__ == '__main__':
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            app.run()
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('tottime')
            stats.print_stats(20)


if __name__ == '__main__':
    main()
