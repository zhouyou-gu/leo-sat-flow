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


def get_relative_lct_view(simulation: GNNSimulation, connected_sat: np.ndarray, connected_lct: np.ndarray, FOR: float):
    satellite_distances = np.linalg.norm(simulation.positions[connected_sat[:, 0]] - simulation.positions[connected_sat[:, 1]], axis=1)
    satellite_distances = satellite_distances * simulation.EARTH_RADIUS
    dir_from_src = (simulation.positions[connected_sat[:, 1]] - simulation.positions[connected_sat[:, 0]]) / np.linalg.norm(simulation.positions[connected_sat[:, 0]] - simulation.positions[connected_sat[:, 1]], axis=1, keepdims=True)
    dir_from_dst = -dir_from_src
    src_mounting_dir = simulation.lct_directions[connected_sat[:, 0], connected_lct[:, 0]%simulation.N_LCT_PER_SAT].reshape(-1,3)
    dst_mounting_dir = simulation.lct_directions[connected_sat[:, 1], connected_lct[:, 1]%simulation.N_LCT_PER_SAT].reshape(-1,3)

    cos_theta_src = np.sum(dir_from_src * src_mounting_dir, axis=1)
    cos_theta_dst = np.sum(dir_from_dst * dst_mounting_dir, axis=1)

    # check all angles are within FOR
    if not (np.all(cos_theta_src >= np.cos(np.radians(FOR))) and np.all(cos_theta_dst >= np.cos(np.radians(FOR)))):
        print("Error: Some connected LCTs exceed the FOR limit!")
        print("error src angles (deg):", np.degrees(np.arccos(cos_theta_src[cos_theta_src < np.cos(np.radians(FOR))])))
        print("error dst angles (deg):", np.degrees(np.arccos(cos_theta_dst[cos_theta_dst < np.cos(np.radians(FOR))])))
    else:
        print("All connected LCTs are within the FOR limit.")
        print("Max src angle (deg):", np.degrees(np.arccos(np.min(cos_theta_src))))
        print("Max dst angle (deg):", np.degrees(np.arccos(np.min(cos_theta_dst))))


    y_dir_list = np.zeros((connected_lct.shape[0],4), dtype=np.int32)
    y_dir_list[:,0] = 2
    y_dir_list[:,1] = 3
    y_dir_list[:,2] = 1
    y_dir_list[:,3] = 0
    sat_pos_dir = simulation.positions/np.linalg.norm(simulation.positions, axis=1, keepdims=True)

    s_y_dir = simulation.lct_directions[connected_sat[:, 0],  y_dir_list[np.arange(y_dir_list.shape[0]),connected_lct[:, 0]%simulation.N_LCT_PER_SAT],:].reshape(-1,3)
    d_y_dir = simulation.lct_directions[connected_sat[:, 1],  y_dir_list[np.arange(y_dir_list.shape[0]),connected_lct[:, 1]%simulation.N_LCT_PER_SAT],:].reshape(-1,3)

    s_x = cos_theta_src
    s_y = np.sum(dir_from_src * s_y_dir, axis=1)
    s_z = np.sum(dir_from_src * sat_pos_dir[connected_sat[:, 0]], axis=1)

    d_x = cos_theta_dst
    d_y = np.sum(dir_from_dst * d_y_dir, axis=1)
    d_z = np.sum(dir_from_dst * sat_pos_dir[connected_sat[:, 1]], axis=1)

    
    ret = np.concatenate((s_x.reshape(-1,1), s_y.reshape(-1,1), s_z.reshape(-1,1),
                          d_x.reshape(-1,1), d_y.reshape(-1,1), d_z.reshape(-1,1)), axis=1)
    
    ret = ret * satellite_distances.reshape(-1,1)
    ret = ret.reshape(-1,3)
    return ret

if __name__ == "__main__":
    N_CONSTELLATION = 1
    SG_STEPS_ALL = 500
    RUN_SG_N_SAT = 2000
    # Load tle data.
    from working_dir_path import get_working_dir_path
    import os
    tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')

    LOG_OBJ = STATS_OBJECT()
    LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

    LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)
    for FOR in [30, 60, 90]:
        for seed in range(N_CONSTELLATION):
            solver = ldl_sg_compare_solver()
            ts, valid_satellites, sat_array = generate_tle_partly_regular_constellation1000(n_sat=2000, ratio=0.0, starlink_tle_path=tle_file_path, seed=seed)
            GNNSimulation.FOR_THETA_HALF = FOR
            simulation = GNNSimulation(ts, sat_array)
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
            
            rates_list = []
            p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = solver.get_prim_objective(with_rates=True)
            relative_views = get_relative_lct_view(simulation, connected_sat, connected_lct, FOR=FOR)
            np.savetxt(os.path.join(LOG_DIR, f"relative_lct_view_for_{FOR}_seed_{seed}_dujo.csv"), relative_views, delimiter=",")
            s_t_rates = np.concatenate((solver.data_source.reshape(-1,1), solver.data_target.reshape(-1,1), rates.reshape(-1,1)), axis=1)
            np.savetxt(os.path.join(LOG_DIR, f"connected_s_t_for_{FOR}_seed_{seed}_dujo.csv"), s_t_rates, delimiter=",")
            rates_list.append(rates.sum())
            
            p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = solver.get_prim_objective_heuristic(with_rates=True, matching_method='mwm')
            relative_views = get_relative_lct_view(simulation, connected_sat, connected_lct, FOR=FOR)
            np.savetxt(os.path.join(LOG_DIR, f"relative_lct_view_for_{FOR}_seed_{seed}_mwm.csv"), relative_views, delimiter=",")
            s_t_rates = np.concatenate((solver.data_source.reshape(-1,1), solver.data_target.reshape(-1,1), rates.reshape(-1,1)), axis=1)
            np.savetxt(os.path.join(LOG_DIR, f"connected_s_t_for_{FOR}_seed_{seed}_mwm.csv"), s_t_rates, delimiter=",")
            rates_list.append(rates.sum())
            
            p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = solver.get_prim_objective_heuristic(with_rates=True, matching_method='rand')
            relative_views = get_relative_lct_view(simulation, connected_sat, connected_lct, FOR=FOR)
            np.savetxt(os.path.join(LOG_DIR, f"relative_lct_view_for_{FOR}_seed_{seed}_rand.csv"), relative_views, delimiter=",")
            s_t_rates = np.concatenate((solver.data_source.reshape(-1,1), solver.data_target.reshape(-1,1), rates.reshape(-1,1)), axis=1)
            np.savetxt(os.path.join(LOG_DIR, f"connected_s_t_for_{FOR}_seed_{seed}_rand.csv"), s_t_rates, delimiter=",")
            rates_list.append(rates.sum())
            
            p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = solver.get_prim_objective_heuristic(with_rates=True, matching_method='grid')
            relative_views = get_relative_lct_view(simulation, connected_sat, connected_lct, FOR=FOR)
            np.savetxt(os.path.join(LOG_DIR, f"relative_lct_view_for_{FOR}_seed_{seed}_grid.csv"), relative_views, delimiter=",")
            s_t_rates = np.concatenate((solver.data_source.reshape(-1,1), solver.data_target.reshape(-1,1), rates.reshape(-1,1)), axis=1)
            np.savetxt(os.path.join(LOG_DIR, f"connected_s_t_for_{FOR}_seed_{seed}_grid.csv"), s_t_rates, delimiter=",")
            rates_list.append(rates.sum())
            
            print(rates_list)
            