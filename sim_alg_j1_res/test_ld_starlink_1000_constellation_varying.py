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
        now = self.ts.now()
        new_tt_jd = now.tt + delta_days
        self.current_time = self.ts.tt(jd=new_tt_jd)
        return self.current_time
        
# Load tle data.
from working_dir_path import get_working_dir_path
import os
tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)

for seed in range(100):
    solver = ldl_sg_compare_solver()
    ts, valid_satellites, sat_array = generate_tle_partly_regular_constellation1000(n_sat=1000, ratio=0.0, starlink_tle_path=tle_file_path, seed=seed)
    simulation = GNNSimulation(ts, sat_array)
    simulation.config_l_mask(seed=seed)
    simulation.update_space()
    simulation.set_solver(solver)
    tic = LOG_OBJ._get_tic()
    for step in range(500):
        print(f"Running simulation with step: {step}")
        ratio, p_o, d_o, p_o_mwm = simulation.run_step()
        tim = LOG_OBJ._get_tim(tic,remove_timer=False)
        LOG_CSV_WRITTER.log_mul_scalar("sg", step, [tim,ratio, p_o, d_o, p_o_mwm], g_step=seed)

    tic = LOG_OBJ._get_tic()
    PATH = "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/train_ld_starlink_1000_varying_beta/train_ld_starlink_1000_varying_beta-2025-September-11-13-34-44-ail/ld_model_target.model_final_beta_0_5000_pt.pt"
    solver.load_gnn(path=PATH)
    solver.infer_gnn()
    d_o, edge_prices = solver.get_dual_objective(with_prices=True)
    p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = solver.get_prim_objective(with_rates=True)
    tim = LOG_OBJ._get_tim(tic,remove_timer=True)
    LOG_CSV_WRITTER.log_mul_scalar("ldl", step, [tim, ratio, p_o, d_o, p_o_mwm], g_step=seed)

    

