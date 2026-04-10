#!/usr/bin/env python3
"""
Simulation of Starlink satellites with Earth rotation and Vispy visualization.

This script loads Starlink TLE data, filters invalid satellites, computes Earth’s rotation,
and visualizes both the Earth (with a textured sphere) and satellites in a 3D scene.
"""

import math
import time
import os

import plotext
import psutil
import numpy as np
from scipy.spatial import cKDTree

from skyfield.api import load
from skyfield.sgp4lib import TEME

from sim_mld.ml.e2e_rl.model import rl_model
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


class pg_gnnsolver(gnnsolver):
    def init_gnn(self, path=None, BETA = 0.5, GAMMA=1):
        print("Initializing GNN model")
        self.model = rl_model(BETA=BETA, GAMMA=GAMMA, DET=False)

if __name__ == "__main__":
    # Load tle data.
    from working_dir_path import get_working_dir_path
    tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')

    LOG_OBJ = STATS_OBJECT()
    LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"LOG_DIR: {LOG_DIR}")

    LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)


    solver = pg_gnnsolver()
    solver.init_gnn(BETA=0.7, GAMMA=1)
    tic = LOG_OBJ._get_tic()
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
        tim = LOG_OBJ._get_tim(tic,remove_timer=False)
        LOG_CSV_WRITTER.log_mul_scalar("res", step, [tim, ratio, p_o, d_o, p_o_mwm])
    solver.model.save(LOG_DIR, "model_final_pg")
    LOG_CSV_WRITTER.close()
