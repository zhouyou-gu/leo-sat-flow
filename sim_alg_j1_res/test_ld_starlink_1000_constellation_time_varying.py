#!/usr/bin/env python3

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

class Time_Varying_Simulation(GNNSimulation):
    def step_time_us(self, time_us):
        delta_days = time_us / 1e6 / 86400  # Convert microseconds to days.
        now = self.current_time
        new_tt_jd = now.tt + delta_days
        self.current_time = self.ts.tt(jd=new_tt_jd)
        return self.current_time
    
    def get_simulation_time(self):
        # return time at 2025 jun 1st
        return self.current_time
    
    def reset_simulation_time(self):
        self.current_time = self.ts.utc(2025, 7, 16, 16, 0, 0)
        
if __name__ == "__main__":
    SG_STEPS = 500
    N_CONSTELLATION = 10
    
    # Load tle data.
    from working_dir_path import get_working_dir_path
    import os
    tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')

    LOG_OBJ = STATS_OBJECT()
    LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

    LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)

    for N_SAT in [1000]:
        for seed in range(N_CONSTELLATION):
            sg_solver = ldl_sg_compare_solver()
            ts, valid_satellites, sat_array = generate_tle_partly_regular_constellation1000(n_sat=N_SAT, ratio=0.0, starlink_tle_path=tle_file_path, seed=seed)
            init_simulation = Time_Varying_Simulation(ts, sat_array)
            init_simulation.reset_simulation_time()
            init_simulation.config_l_mask(seed=seed)
            init_simulation.update_space()
            init_simulation.set_solver(sg_solver)
            
            varying_sg_simulation = Time_Varying_Simulation(ts, sat_array)
            varying_sg_simulation.reset_simulation_time()
            varying_sg_simulation.config_l_mask(seed=seed)
            varying_sg_simulation.update_space()

            tic = LOG_OBJ._get_tic()
            for step in range(SG_STEPS):
                print(f"Running simulation with step: {step}")
                ratio, p_o, d_o, p_o_mwm = init_simulation.run_step()
                tim = LOG_OBJ._get_tim(tic,remove_timer=False)
                varying_sg_simulation.reset_simulation_time()
                varying_sg_simulation.step_time_us(tim)
                varying_sg_simulation.update_space()
                tmp_solver = ldl_sg_compare_solver()
                varying_sg_simulation.set_solver(tmp_solver)
                tmp_solver.price_graph.price_graph = sg_solver.price_graph.price_graph.copy()
                varying_sg_simulation.update_solver_traffic_info(seed=seed)
                p_o_delayed = tmp_solver.get_prim_objective(with_rates=False)
                p_o_instant = sg_solver.get_prim_objective(with_rates=False)
                print(f"Instant p_o: {p_o_instant}, delayed p_o: {p_o_delayed}")
                LOG_CSV_WRITTER.log_mul_scalar("sg", step, [tim, p_o_delayed, p_o_instant, N_SAT], g_step=seed)


            varying_gnn_simulation = Time_Varying_Simulation(ts, sat_array)
            varying_gnn_simulation.reset_simulation_time()
            varying_gnn_simulation.config_l_mask(seed=seed)
            varying_gnn_simulation.update_space()
            
            gnn_solver = ldl_sg_compare_solver()
            init_simulation = Time_Varying_Simulation(ts, sat_array)
            init_simulation.reset_simulation_time()
            init_simulation.config_l_mask(seed=seed)
            init_simulation.update_space()
            init_simulation.set_solver(gnn_solver)
            init_simulation.update_solver_traffic_info(seed=seed)

            PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "selected_nn/ld_model.model_final_beta_0_7000_pt.pt")
    
            gnn_solver.load_gnn(path=PATH)
            tic = LOG_OBJ._get_tic()
            gnn_solver.infer_gnn()
            p_o_gnn_instant = gnn_solver.get_prim_objective(with_rates=False)
            tim = LOG_OBJ._get_tim(tic,remove_timer=True)
            varying_gnn_simulation.step_time_us(0)
            varying_gnn_simulation.update_space()
            
            tmp_solver = ldl_sg_compare_solver()
        
            varying_gnn_simulation.set_solver(tmp_solver)
            varying_gnn_simulation.update_solver_traffic_info(seed=seed)
            tmp_solver.price_graph.price_graph = gnn_solver.price_graph.price_graph.copy()

            p_o_gnn_delayed = tmp_solver.get_prim_objective(with_rates=False)
            LOG_CSV_WRITTER.log_mul_scalar("ldl", step, [tim, p_o_gnn_delayed, p_o_gnn_instant, N_SAT], g_step=seed)
            print(f"GNN delayed p_o: {p_o_gnn_delayed}")
            varying_heu_simulation = Time_Varying_Simulation(ts, sat_array)
            varying_heu_simulation.reset_simulation_time()
            varying_heu_simulation.config_l_mask(seed=seed)
            varying_heu_simulation.update_space()
            
            heu_solver = ldl_sg_compare_solver()
            init_simulation = Time_Varying_Simulation(ts, sat_array)
            init_simulation.reset_simulation_time()
            init_simulation.config_l_mask(seed=seed)
            init_simulation.update_space()
            init_simulation.set_solver(heu_solver)
            init_simulation.update_solver_traffic_info(seed=seed)
            
            tic = LOG_OBJ._get_tic()
            p_o_heu = heu_solver.get_prim_objective_heuristic(with_rates=False)
            tim = LOG_OBJ._get_tim(tic,remove_timer=True)

            LOG_CSV_WRITTER.log_mul_scalar("heu", step, [tim, p_o_heu, N_SAT], g_step=seed)

