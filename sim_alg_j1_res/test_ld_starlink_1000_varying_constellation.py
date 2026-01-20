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

from sim_alg_j1_res.train_rl_starlink_1000_ld import GNNSimulation

from sim_alg_j1_res.test_ld_starlink_1000_sg_compare import ldl_sg_compare_solver


class kuiper_simulation(GNNSimulation):
    def get_simulation_time(self):
        return self.ts.utc(2026, 1, 20, 11, 0, 0)



if __name__ == "__main__":
    N_CONSTELLATION = 2
    SG_STEPS_ALL = 500
    # Load tle data.
    from working_dir_path import get_working_dir_path
    import os
    starlink_tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')
    oneweb_tle_file_path = os.path.join(get_working_dir_path(),'oneweb_16_jul_2025_1600.tle')
    kuiper_tle_file_path = os.path.join(get_working_dir_path(),'kuiper_20_jan_2026_1100.tle')

    LOG_OBJ = STATS_OBJECT()
    LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

    LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)
    for constellation in ["starlink", "walker-delta", "oneweb", "kuiper"]:
        for seed in range(N_CONSTELLATION):
            if constellation == "oneweb":
                ts, valid_satellites, sat_array = load_url_tle_data(oneweb_tle_file_path, reload=True)
            elif constellation == "kuiper":
                ts, valid_satellites, sat_array = load_url_tle_data(kuiper_tle_file_path, reload=True)
            elif constellation == "starlink":
                ts, valid_satellites, sat_array = generate_tle_partly_regular_constellation1000(n_sat=1000, ratio=0., starlink_tle_path=starlink_tle_file_path, seed=seed)
            elif constellation == "walker-delta":
                ts, valid_satellites, sat_array = generate_tle_partly_regular_constellation1000(n_sat=1000, ratio=1., starlink_tle_path=starlink_tle_file_path, seed=seed)

            solver = ldl_sg_compare_solver()
            simulation = GNNSimulation(ts, sat_array)
            if constellation == "kuiper":
                simulation = kuiper_simulation(ts, sat_array)
            if constellation == "oneweb" or constellation == "kuiper":
                simulation.terrain.GW_RANGE = 1800.0  # in kilometers, range of the ground station
                simulation.LISL_MAX_DISTANCE = 4000.0  # in kilometers, max distance for laser links  
            simulation.config_l_mask(seed=seed)
            simulation.update_space()
            simulation.set_solver(solver)
            simulation.update_solver_traffic_info(seed=0)
            # simulation.solver.update_step_rates_prices()


            PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "selected_nn/ld_model.model_final_beta_0_7000_pt.pt")
            solver.load_gnn(path=PATH)
            tic = LOG_OBJ._get_tic()
            solver.infer_gnn()
            tim = LOG_OBJ._get_tim(tic,remove_timer=True)
            d_o, edge_prices = solver.get_dual_objective(with_prices=True)
            p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = solver.get_prim_objective(with_rates=True)
            p_o_mwm = solver.get_prim_objective_heuristic(with_rates=False, matching_method='mwm')
            p_o_random = solver.get_prim_objective_heuristic(with_rates=False, matching_method='rand')
            p_o_grid = solver.get_prim_objective_heuristic(with_rates=False, matching_method='grid')
            p_o_sate = solver.get_prim_objective_heuristic(with_rates=False, matching_method='mwm', routing_method='spf')

            LOG_CSV_WRITTER.log_mul_scalar("ldl", 0, [tim, 1., p_o, d_o, p_o_mwm, p_o_random, p_o_grid, p_o_sate], g_step=seed)
