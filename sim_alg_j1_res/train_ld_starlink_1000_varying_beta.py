#!/usr/bin/env python3
"""
Simulation of Starlink satellites with Earth rotation and Vispy visualization.

This script loads Starlink TLE data, filters invalid satellites, computes Earth’s rotation,
and visualizes both the Earth (with a textured sphere) and satellites in a 3D scene.
"""

import math
import time

import plotext
import psutil
import numpy as np
from scipy.spatial import cKDTree

from skyfield.api import load
from skyfield.sgp4lib import TEME

from sim_mld.ml.ld_sgl.model import ld_model
from sim_mld.solver import *
from sim_mld.visual import *
from sim_mld.tle import *
from sim_mld.constellation import *
from sim_mld.simulation import Simulation

from vispy import app
import logging

from sim_src.util import CSV_WRITER_OBJECT, GET_LOG_PATH_FOR_SIM_SCRIPT, STATS_OBJECT, counted
from working_dir_path import get_working_dir_path

import torch
torch.set_float32_matmul_precision('medium')

np.set_printoptions(precision=4, suppress=True)

from sim_alg_j1_res.train_rl_starlink_1000_ld import gnnsolver, GNNSimulation
      
# Load tle data.
from working_dir_path import get_working_dir_path
import os
tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)


for BETA in [0.5, 0.7, 0.9]:
    solver = gnnsolver()
    solver.init_gnn(BETA=BETA, GAMMA=1)
    for step in range(500):
        print(f"Running simulation with step: {step}")
        # Create simulation instance.
        ts, valid_satellites, sat_array = generate_tle_partly_regular_constellation1000(n_sat=1000, ratio=0.0, starlink_tle_path=tle_file_path, seed=step)
        GNNSimulation.FOR_THETA_HALF = 30.0  # Set the angle for LT direction.
        simulation = GNNSimulation(ts, sat_array)
        simulation.config_l_mask(seed=step)

        simulation.update_space()
        simulation.set_solver(solver)
        ratio, p_o, d_o, p_o_mwm = simulation.run_step()
        BETA_TEXT = f"BETA_{BETA:.4f}".replace('.','_')
        LOG_CSV_WRITTER.log_mul_scalar(BETA_TEXT, step, [ratio, p_o, d_o, p_o_mwm])
        print(f"Finished simulation with step: {step}, BETA: {BETA:.4f}")
    
    solver.model.save(LOG_DIR, f"model_final_beta_{BETA:.4f}.pt".replace('.','_'))
