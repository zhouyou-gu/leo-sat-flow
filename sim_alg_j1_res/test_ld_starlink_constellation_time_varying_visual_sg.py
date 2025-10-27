#!/usr/bin/env python3

import math
import time
from turtle import width

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
import vispy.io as io
from vispy.gloo.util import _screenshot

import logging

from sim_src.util import CSV_WRITER_OBJECT, GET_LOG_PATH_FOR_SIM_SCRIPT, STATS_OBJECT, counted, GET_FILE_NAME_FOR_SIM_SCRIPT
from working_dir_path import get_working_dir_path

import torch
torch.set_float32_matmul_precision('medium')

np.set_printoptions(precision=4, suppress=True)

from sim_alg_j1_res.test_ld_starlink_1000_sg_compare import ldl_sg_compare_solver

from sim_alg_j1_res.test_ld_starlink_constellation_time_varying_visual_ldl import Visual_Time_Varying_Simulation

if __name__ == "__main__":
    SG_STEPS = 100
    N_CONSTELLATION = 1
    
    # Load tle data.
    from working_dir_path import get_working_dir_path
    import os
    tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')

    LOG_OBJ = STATS_OBJECT()
    LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

    LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)


    for seed in range(N_CONSTELLATION):
        solver = ldl_sg_compare_solver()
        ts, valid_satellites, sat_array = generate_tle_partly_regular_constellation1000(n_sat=1000, ratio=0.0, starlink_tle_path=tle_file_path, seed=seed)
        init_simulation = Visual_Time_Varying_Simulation(ts, sat_array)
        init_simulation.setup_visualization()
        init_simulation.reset_simulation_time()
        init_simulation.config_l_mask(seed=seed)
        init_simulation.update_space()
        init_simulation.set_solver(solver)
        init_simulation.update_solver_traffic_info(seed=seed)
        
        tic = LOG_OBJ._get_tic()
        for step in range(SG_STEPS):
            print(f"Running simulation with step: {step}")
            ratio, p_o, d_o, p_o_mwm = init_simulation.run_step()
        tim = LOG_OBJ._get_tim(tic)

        p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = solver.get_prim_objective(with_rates=True)
        active_indicators = np.ones(connected_lct.shape[0], dtype=bool)
        init_simulation.update_connected_lct_pairs(connected_lct, active_indicators)
        connectable_lct_pairs = solver.possible_lct_pair_expanded
        active_indicators = np.ones(connectable_lct_pairs.shape[0], dtype=bool )
        init_simulation.update_connectable_lct_pairs(connectable_lct_pairs, active_indicators)
        init_simulation.update_for()
        init_simulation.camera_rotate_to_0_lat()
        init_simulation.visualize()
        init_simulation.save_img(LOG_DIR)
        

        init_simulation.N_STEP += 1
        init_simulation.step_time_us(tim)
        init_simulation.update_space()
        init_simulation.set_solver(solver)
        init_simulation.update_solver_traffic_info(seed=seed)

        # init_simulation.setup_visualization()
        connectable_lct_pairs_changed = solver.possible_lct_pair_expanded
        mask = (connectable_lct_pairs[:, None, :] == connectable_lct_pairs_changed[None, :, :]).all(-1).any(1)
        init_simulation.update_connectable_lct_pairs(connectable_lct_pairs, mask)

        mask = (connected_lct[:, None, :] ==  connectable_lct_pairs_changed[None, :, :]).all(-1).any(1)
        init_simulation.update_connected_lct_pairs(connected_lct, mask)
        init_simulation.update_for()
        init_simulation.camera_rotate_to_0_lat()
        init_simulation.visualize()
        init_simulation.save_img(LOG_DIR)
        
        init_simulation.close_app()
        