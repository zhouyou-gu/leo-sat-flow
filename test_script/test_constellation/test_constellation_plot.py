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

from PIL import Image
import numpy as np
import simpy
from skyfield.api import load
from skyfield.sgp4lib import TEME
from sgp4.api import SatrecArray
from vispy import app, scene
from vispy.geometry import MeshData, create_sphere
from vispy.visuals.filters import TextureFilter
from vispy.visuals.transforms import MatrixTransform, STTransform

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


def load_starlink_data(url: str):
    """
    Load Starlink TLE data from the provided URL.

    Parameters:
        url (str): URL to the TLE file.

    Returns:
        tuple: (timescale, valid_satellites, sat_array)
    """
    ts = load.timescale()
    satellites = load.tle_file(url)
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
    view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, elevation=45, distance=2)

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
        pos=vector, color='blue', width=3, arrow_size=20, arrow_type='stealth', parent=view.scene
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

    # Set up transformation for the sphere.
    transform = MatrixTransform()
    sphere_visual.transform = transform

    return {
        "canvas": canvas,
        "view": view,
        "sphere_visual": sphere_visual,
        "scatter": scatter,
    }


class Simulation:
    def __init__(self, ts, sat_array, sphere_visual, scatter, canvas):
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
        self.sat_array = sat_array
        self.sphere_visual = sphere_visual
        self.scatter = scatter
        self.canvas = canvas
        self.angle = self.compute_initial_rotation()
        self.update_count = 0
        self.accumulated_update_time = 0

    def compute_initial_rotation(self) -> float:
        """
        Compute the initial rotation angle from the current GMST.

        Returns:
            float: Rotation angle in degrees.
        """
        t_now = self.ts.now()
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

        try:
            current_time = self.ts.now()
            error_upd, positions_upd, velocities_upd = self.sat_array.sgp4(
                np.array([current_time.whole]),
                np.array([current_time.ut1_fraction])
            )
            positions = np.array(positions_upd).reshape(-1, 3)
            # Normalize positions by Earth's radius (6371 km)
            positions = positions / 6371

            # Compute transformation from TEME to ICRS.
            R_icrs_to_teme = TEME.rotation_at(current_time)
            R_teme_to_icrs = R_icrs_to_teme.T
            positions = positions @ R_teme_to_icrs
            color = np.abs(positions)/2
            self.scatter.set_data(positions, face_color=color, size=7, edge_width=0)

            # Reset and apply rotation transformation to the Earth sphere.
            self.sphere_visual.transform.reset()
            self.sphere_visual.transform.rotate(self.compute_initial_rotation(), (0, 0, 1))
        except simpy.core.EmptySchedule as e:
            logger.warning("Empty schedule encountered during update: %s", e)
        except Exception as e:
            logger.error("Unexpected error during update: %s", e)

        # Trigger a canvas update.
        self.canvas.update()

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        self.update_count += 1
        self.accumulated_update_time += elapsed

        if self.update_count % 60 == 0:
            avg_time = self.accumulated_update_time / 60
            logger.info("Average update time over last 60 frames: %.6f seconds", avg_time)
            self.accumulated_update_time = 0


def main():
    # Load Starlink data.
    starlink_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
    ts, valid_satellites, sat_array = load_starlink_data(starlink_url)

    # Set up visualization.
    viz = setup_visualization()
    canvas = viz["canvas"]
    sphere_visual = viz["sphere_visual"]
    scatter = viz["scatter"]

    # Create simulation instance.
    simulation = Simulation(ts, sat_array, sphere_visual, scatter, canvas)

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
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats(20)


if __name__ == '__main__':
    main()
