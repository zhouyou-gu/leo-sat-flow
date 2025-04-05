#!/usr/bin/env python3
"""
Simulation of Starlink satellites with Earth rotation and Vispy visualization.

This script loads Starlink TLE data, filters invalid satellites, computes Earth’s rotation,
and visualizes both the Earth (with a textured sphere) and satellites in a 3D scene.
"""
import psutil

import math
import time
import cProfile
import pstats
import logging
from itertools import combinations

from PIL import Image
import numpy as np
from numba import njit
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


@njit
def greedy_max_weight_matching(E):
    """
    Compute a greedy heuristic maximum weight matching for a NumPy array of edges.
    
    Parameters:
    - E: NumPy array with shape (num_edges, 3) where each row is [u, v, weight]
    
    Returns:
    - matching: List of tuples (u, v, weight) representing the selected edges.
    """
    # Sort edges in descending order by weight using numpy's argsort
    sorted_indices = np.argsort(-E[:, 2])
    E_sorted = E[sorted_indices]
    
    matching = []
    matched_nodes = set()
    
    # Iterate over sorted edges
    for edge in E_sorted:
        u, v, weight = edge
        # Convert u, v to Python ints for set membership checks
        u, v = int(u), int(v)
        if u not in matched_nodes and v not in matched_nodes:
            matching.append((u, v))
            matched_nodes.add(u)
            matched_nodes.add(v)
    
    return matching

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_face_coor(vertices: np.ndarray) -> list:
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


def compute_texcoords(vertices: np.ndarray) -> np.ndarray:
    """
    Compute texture coordinates for all faces from vertices.

    Parameters:
        vertices (np.ndarray): Array where each element represents a face's vertices.

    Returns:
        np.ndarray: Array of texture coordinates.
    """
    texcoords = []
    for face in vertices:
        face_texcoords = compute_face_coor(face)
        texcoords.extend(face_texcoords)
    return np.array(texcoords)


def load_starlink_data(url: str, reload: bool = True) -> tuple:
    """
    Load Starlink TLE data from the provided URL.

    Parameters:
        url (str): URL to the TLE file.

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
    # view.camera = scene.cameras.ArcballCamera(fov=45, distance=2)
    
    # Load texture image for the Earth sphere.
    texture_path = "land_sea_texture.png"
    try:
        texture_image = Image.open(texture_path)
    except Exception as e:
        logger.error("Error loading texture image: %s", e)
        raise

    texture = np.array(texture_image)
    # Duplicate the texture horizontally if needed.
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
    view.add(sphere_visual)

    # Add an arrow visual.
    vector = np.array([[0, 0, 0], [1, 1, 1]])
    arrow = scene.visuals.Arrow(
        pos=vector, color='black', width=3, arrow_size=20, arrow_type='stealth', parent=view.scene
    )
    view.add(arrow)

    # Add axes with scaling.
    axes = scene.visuals.XYZAxis(parent=view.scene)
    view.add(axes)
    scale_factors = (2, 2, 2)
    axes.transform = STTransform(scale=scale_factors)

    # Create markers for satellite positions.
    scatter = scene.visuals.Markers()
    view.add(scatter)

    # Create lines for the satellite LISL.
    arrow = scene.visuals.Arrow()
    view.add(arrow)

    # Create lines for the satellite LISL.
    p_lisl = scene.visuals.Arrow()
    view.add(p_lisl)

    # Create lines for connected LISL
    c_lisl = scene.visuals.Arrow()
    view.add(c_lisl)
    # Set up transformation for the sphere.
    transform = MatrixTransform()
    sphere_visual.transform = transform



    return {
        "canvas": canvas,
        "view": view,
        "sphere_visual": sphere_visual,
        "scatter": scatter,
        "arrow": arrow,
        "p_lisl": p_lisl,
        "c_lisl": c_lisl,
    }


def expand_edges_with_original(edges,repeat_per_node=4):
    # Precompute the grid of offsets with the second node's offset iterating faster
    grid = np.stack(np.meshgrid(np.arange(repeat_per_node), np.arange(repeat_per_node), indexing='ij'), axis=-1).reshape(-1, 2)
    
    # For each edge, compute the expanded edges using broadcasting:
    # Each edge is scaled by 4 and then added with every combination of offsets in grid.
    expanded_edges = (edges[:, None, :] * repeat_per_node + grid[None, :, :]).reshape(-1, 2)
    
    # Repeat each original edge 16 times so that each expanded edge corresponds to an original edge.
    repeated_original = np.repeat(edges, repeat_per_node*repeat_per_node, axis=0)
    
    return repeated_original, expanded_edges

class Simulation:
    FOR_THETA = 15.0  # Angle in degrees for the satellite LT direction
    LISL_MAX_DISTANCE = 3000.0  # Maximum distance for LISL in km
    TIME_SCALE = 10.0
    EARTH_RADIUS = 6371.0  # Earth's radius in km
    
    FRONT_COLOR = np.array([0, 0, 1, 1])
    BACK_COLOR = np.array([1, 0.75, 0, 1])
    RIGHT_COLOR = np.array([1, 0, 0, 1])
    LEFT_COLOR = np.array([0.05, 1, 0.05, 1])
    
    def __init__(self, ts, sat_array, sphere_visual, scatter, arrow, p_lisl, c_lisl, canvas):
        """
        Initialize the simulation.

        Parameters:
            ts: Skyfield timescale.
            sat_array: Vectorized satellite propagation array.
            sphere_visual: Vispy visual for the Earth sphere.
            scatter: Vispy visual for satellite markers.
            canvas: Vispy SceneCanvas.
        """
        self.ts = ts

        # Store the simulation start time (Skyfield Time) and the real start time.
        self.simulation_start_time = self.ts.now()
        self.real_start_time = time.perf_counter()
        
        self.sat_array = sat_array
        self.sphere_visual = sphere_visual
        self.scatter = scatter
        self.arrow = arrow
        self.p_lisl = p_lisl
        self.c_lisl = c_lisl
        self.canvas = canvas
        self.angle = self.compute_rotation()
        self.update_count = 0
        self.accumulated_update_time = 0

    def get_simulation_time(self):
        """
        Compute the current simulation time based on the time scaling factor.

        Returns:
            Skyfield Time: The current simulation time.
        """
        # Compute elapsed real time in seconds
        elapsed_real = time.perf_counter() - self.real_start_time
        # Apply the time scaling factor (convert seconds to days, as Skyfield times are in Julian days)
        elapsed_scaled = elapsed_real * self.TIME_SCALE
        delta_days = elapsed_scaled / 86400  # Convert seconds to days

        # Compute the new simulation time
        new_tt_jd = self.simulation_start_time.tt + delta_days
        sim_time = self.ts.tt(jd=new_tt_jd)
        return sim_time


    def compute_rotation(self) -> float:
        """
        Compute the initial rotation angle from the current GMST.

        Returns:
            float: Rotation angle in degrees.
        """
        t_now = self.get_simulation_time()
        gmst_hours = t_now.gmst
        rotation_angle_deg = gmst_hours * 15  # 24h -> 360° (15° per hour)
        logger.debug("GMST: %.2f hours, Rotation angle: %.2f degrees", gmst_hours, rotation_angle_deg)
        return rotation_angle_deg

    def update(self, event):
        """
        Update function called on each timer tick to update the simulation.
        """
        start_time = time.perf_counter()
        self.sphere_visual.update()

        # Get the current CPU usage percentage over a 1-second interval.
        cpu_usage = psutil.cpu_percent()

        # Get virtual memory statistics.
        mem = psutil.Process().memory_info().rss

        print(f"CPU Usage: {cpu_usage}%, Memory Usage: {mem / 1e6} MB")
        try:
            # Reset and apply rotation transformation to the Earth sphere.
            self.sphere_visual.transform.reset()
            self.sphere_visual.transform.rotate(self.compute_rotation(), (0, 0, 1))
            
            # Get the current simulation time and propagate the satellites.
            current_time = self.get_simulation_time()
            error_upd, positions_upd, velocities_upd = self.sat_array.sgp4(
                np.array([current_time.whole]),
                np.array([current_time.ut1_fraction])
            )
            positions = np.array(positions_upd).reshape(-1, 3)
            velocities = np.array(velocities_upd).reshape(-1, 3)
            # Normalize positions by Earth's radius (6371 km)
            positions = positions / self.EARTH_RADIUS
            velocities = velocities / self.EARTH_RADIUS
            
            # Compute transformation from TEME to ICRS.
            R_icrs_to_teme = TEME.rotation_at(current_time)
            R_teme_to_icrs = R_icrs_to_teme.T
            positions = positions @ R_teme_to_icrs
            velocities = velocities @ R_teme_to_icrs
            
            # self.scatter.set_data(positions, face_color=np.abs(positions)/2, size=7, edge_width=0)
            self.scatter.set_data(positions, face_color=[0,0,0,0.5], size=10, edge_width=0)

            # Create lines for the satellite LT directions.
            front = velocities/np.linalg.norm(velocities, axis=1)[:, np.newaxis] 
            back  = -front
            down  = positions/np.linalg.norm(positions, axis=1)[:, np.newaxis]
            right = np.cross(down,front)        
            left = -right
            
            a_from = np.tile(positions, (4,1))
            a_to = np.concatenate((front, back, right, left), axis=0)*0.01 + a_from
            
            a_data = np.concatenate((a_from, a_to), axis=1)
            a_data = a_data.reshape(-1, 3)

            rows = positions.shape[0]*8
            arrow_color = np.zeros((rows, 4))
            arrow_color[0:rows // 4, :] = self.FRONT_COLOR
            arrow_color[rows // 4:rows // 2, :] = self.BACK_COLOR
            arrow_color[rows // 2:3 * rows // 4, :] = self.RIGHT_COLOR
            arrow_color[3 * rows // 4:, :] = self.LEFT_COLOR
            self.arrow.set_data(pos=a_data, \
                                color=arrow_color, width=5,connect='segments')
            
            if True:
                tree_time = time.perf_counter()

                # Build the k-d tree
                tree = cKDTree(positions)
                # Get all pairs within the threshold
                distances = self.LISL_MAX_DISTANCE/self.EARTH_RADIUS
                edges = tree.query_pairs(r=distances,output_type='ndarray')
                # data = np.ones(len(edges), dtype=int)

                sat_p_i = positions[edges[:,0]]
                sat_p_j = positions[edges[:,1]]
                
                dir = sat_p_j - sat_p_i
                dir = dir/np.linalg.norm(dir, axis=1)[:, np.newaxis]
                view_from_stack  = np.concatenate((   np.einsum('ij,ij->i', front[edges[:,0]], dir).reshape(-1,1), \
                                                np.einsum('ij,ij->i', back[edges[:,0]], dir).reshape(-1,1), \
                                                np.einsum('ij,ij->i', right[edges[:,0]], dir).reshape(-1,1), \
                                                np.einsum('ij,ij->i', left[edges[:,0]], dir).reshape(-1,1)), axis=1)
                i_j_stack = view_from_stack > math.cos(math.radians(self.FOR_THETA))
                i_j_binary_indicator = i_j_stack.sum(axis=1) > 0
                
                view_to_stack  = np.concatenate((   np.einsum('ij,ij->i', front[edges[:,1]], -dir).reshape(-1,1), \
                                                np.einsum('ij,ij->i', back[edges[:,1]], -dir).reshape(-1,1), \
                                                np.einsum('ij,ij->i', right[edges[:,1]], -dir).reshape(-1,1), \
                                                np.einsum('ij,ij->i', left[edges[:,1]], -dir).reshape(-1,1)), axis=1)
                j_i_stack = view_to_stack > math.cos(math.radians(self.FOR_THETA))
                j_i_binary_indicator = j_i_stack.sum(axis=1) > 0
                
                final_binary_indicator = np.logical_and(i_j_binary_indicator, j_i_binary_indicator)
                possible_edges_between_sat = edges[final_binary_indicator]            
                
                p_lisl_LT_pair = np.einsum('ij,ik->ijk', i_j_stack[final_binary_indicator], j_i_stack[final_binary_indicator]).reshape(-1)
                
                view_from_stack = view_from_stack[final_binary_indicator]
                view_to_stack = view_to_stack[final_binary_indicator]
                view_LT_pair = np.minimum(view_from_stack[:, :, None], view_to_stack[:, None, :]).reshape(-1)
                view_LT_pair = view_LT_pair[p_lisl_LT_pair].reshape(-1, 1)

                repeated_edges, expanded_edges_i_j = expand_edges_with_original(possible_edges_between_sat)

                repeated_edges = repeated_edges[p_lisl_LT_pair]
                expanded_edges_i_j = expanded_edges_i_j[p_lisl_LT_pair]

                edges_color = np.concatenate((self.FRONT_COLOR[np.newaxis], self.BACK_COLOR[np.newaxis], self.RIGHT_COLOR[np.newaxis], self.LEFT_COLOR[np.newaxis]), axis=0)
                expanded_edges_from = expanded_edges_i_j[:,0] % 4
                expanded_edges_to = expanded_edges_i_j[:,1] % 4
                edges_color_from = np.concatenate((edges_color[expanded_edges_from], edges_color[expanded_edges_from]), axis=1).reshape(-1, 4)
                edges_color_to = np.concatenate((edges_color[expanded_edges_to], edges_color[expanded_edges_to]), axis=1).reshape(-1, 4)
                edges_color_data = np.concatenate((edges_color_from, edges_color_to), axis=0)            
                edges_color_data[:, 3] = 0.005
                
                p_from = positions[repeated_edges[:,0]]
                p_to = positions[repeated_edges[:,1]]
                p_mid = (p_from + p_to) / 2
                p_lisl_data_from = np.concatenate((p_from, p_mid), axis=1).reshape(-1, 3)
                p_lisl_data_to = np.concatenate((p_mid, p_to), axis=1).reshape(-1, 3)
                

                p_lisl_data = np.concatenate((p_lisl_data_from, p_lisl_data_to), axis=0)
                # self.p_lisl.set_data(pos=p_lisl_data, \
                #                     color=edges_color_data, width=0.75,connect='segments')
                
            if True:
                relative_speed = velocities[repeated_edges[:,1]] - velocities[repeated_edges[:,0]]
                relative_direction = positions[repeated_edges[:,1]] - positions[repeated_edges[:,0]]
                cross = np.cross(relative_speed, relative_direction)
                angluar_speed = np.linalg.norm(cross, axis=1).reshape(-1, 1)/np.linalg.norm(relative_direction, axis=1).reshape(-1, 1)
                view_time_approx = (math.radians(self.FOR_THETA)-np.acos(view_LT_pair)) / np.abs(angluar_speed)

                included_edges = view_time_approx.reshape(-1) > 100
                
                weighted_edges_i_j = np.concatenate((expanded_edges_i_j[included_edges], view_time_approx[included_edges]), axis=1)
               
                matching = greedy_max_weight_matching(weighted_edges_i_j)
                
                # G = nx.Graph()
                # G.add_weighted_edges_from(weighted_edges_i_j)     
                # matching = nx.algorithms.matching.maximal_matching(G)
                
                m = len(matching)
                # Alternatively, more directly:
                flat_iter = ((min(e), max(e)) for e in matching)
                # Use fromiter to create a flat array of length 2*m
                flat_array = np.fromiter((x for pair in flat_iter for x in pair), dtype=int, count=2*m)

                connected_lisl_edges = flat_array.reshape(-1, 2)
                connected_sat = connected_lisl_edges // 4
                
                edges_color = np.concatenate((self.FRONT_COLOR[np.newaxis], self.BACK_COLOR[np.newaxis], self.RIGHT_COLOR[np.newaxis], self.LEFT_COLOR[np.newaxis]), axis=0)
                expanded_edges_from = connected_lisl_edges[:,0] % 4
                expanded_edges_to = connected_lisl_edges[:,1] % 4
                edges_color_from = np.concatenate((edges_color[expanded_edges_from], edges_color[expanded_edges_from]), axis=1).reshape(-1, 4)
                edges_color_to = np.concatenate((edges_color[expanded_edges_to], edges_color[expanded_edges_to]), axis=1).reshape(-1, 4)
                edges_color_data = np.concatenate((edges_color_from, edges_color_to), axis=0)            
                edges_color_data[:, 3] = 0.75
                
                p_from = positions[connected_sat[:,0]]*1.0001
                p_to = positions[connected_sat[:,1]]*1.0001
                p_mid = (p_from + p_to) / 2
                p_lisl_data_from = np.concatenate((p_from, p_mid), axis=1).reshape(-1, 3)
                p_lisl_data_to = np.concatenate((p_mid, p_to), axis=1).reshape(-1, 3)
                
                c_lisl_data = np.concatenate((p_lisl_data_from, p_lisl_data_to), axis=0)
                self.c_lisl.set_data(pos=c_lisl_data, \
                                    color=edges_color_data, width=1.5,connect='segments')
   
            tree_end_time = time.perf_counter()

        except Exception as e:
            logger.error("Unexpected error during update: %s", e)

        # Trigger a canvas update.
        self.canvas.update()

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        self.update_count += 1
        self.accumulated_update_time += elapsed

        if self.update_count % 5 == 0:
            avg_time = self.accumulated_update_time / 60
            logger.info("Average update time over last 5 frames: %.6f seconds", avg_time)
            self.accumulated_update_time = 0
            logger.info("KDTree build time: %.6f seconds", tree_end_time - tree_time)
            logger.info("Number of edges: %d", edges.shape[0])
            logger.info("Number of possible edges: %d", possible_edges_between_sat.shape[0])
            logger.info("Number of LISL edges: %d", expanded_edges_i_j.shape[0])


def main():
    # Load Starlink data.
    satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
    # satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle'
    ts, valid_satellites, sat_array = load_starlink_data(satellite_url,reload=False)

    # Set up visualization.
    viz = setup_visualization()
    canvas = viz["canvas"]
    sphere_visual = viz["sphere_visual"]
    scatter = viz["scatter"]
    arrow = viz["arrow"]
    p_lisl = viz["p_lisl"]
    c_lisl = viz["c_lisl"]
    # Create simulation instance.
    simulation = Simulation(ts, sat_array, sphere_visual, scatter, arrow, p_lisl, c_lisl, canvas)

    # Set up a timer to update the simulation at roughly 60 FPS.
    timer = app.Timer(interval=1 / 60.0, connect=simulation.update, start=True)

    # Run the application with optional profiling.
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
