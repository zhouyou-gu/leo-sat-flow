#!/usr/bin/env python3
"""
Simulation of Starlink satellites with Earth rotation and Vispy visualization.

This script loads Starlink TLE data, filters invalid satellites, computes Earth’s rotation,
and visualizes both the Earth (with a textured sphere) and satellites in a 3D scene.
"""

import math
import time

import psutil
import numpy as np
from scipy.spatial import cKDTree

from skyfield.api import load
from skyfield.sgp4lib import TEME

from sim_alg.visual import *
from sim_alg.tle import *
from sim_alg.constellation import *
from sim_alg.simulation import Simulation

from vispy import app
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.set_printoptions(precision=3, suppress=True)

if __name__ == '__main__':
    logger.level = logging.DEBUG
    # Load tle data.
    ts, valid_satellites, sat_array = generate_walker_constellation_add_planes()

    # Create simulation instance.
    simulation = Simulation(ts, sat_array)

    # Set up a timer to update the simulation at roughly 60 FPS.
    timer1 = app.Timer(interval=0.0001, connect=simulation.update, iterations=10000, start=True)
    timer2 = app.Timer(interval=1/60., start=True)
    app.run()
