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

class kuiper_simulation(Time_Varying_Simulation):
    def reset_simulation_time(self):
        self.current_time = self.ts.utc(2026, 1, 20, 11, 0, 0)
    

if __name__ == "__main__":
    N_CONSTELLATION = 1
    
    # Load tle data.
    from working_dir_path import get_working_dir_path
    import os

    starlink_tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')
    oneweb_tle_file_path = os.path.join(get_working_dir_path(),'oneweb_16_jul_2025_1600.tle')
    kuiper_tle_file_path = os.path.join(get_working_dir_path(),'kuiper_20_jan_2026_1100.tle')

    LOG_OBJ = STATS_OBJECT()
    LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

    LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)

    N_SAT = 1000
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



            sg_solver = ldl_sg_compare_solver()
            if constellation == "kuiper":
                init_simulation = kuiper_simulation(ts, sat_array)
            else:
                init_simulation = Time_Varying_Simulation(ts, sat_array)
            if constellation == "oneweb" or constellation == "kuiper":
                init_simulation.terrain.GW_RANGE = 1800.0  # in kilometers, range of the ground station
                init_simulation.LISL_MAX_DISTANCE = 4000.0  # in kilometers, max distance for laser links  
            init_simulation.reset_simulation_time()
            init_simulation.config_l_mask(seed=seed)
            init_simulation.update_space()
            init_simulation.set_solver(sg_solver)
            
            connectable_lct_pairs = init_simulation.filtered_lct_pair_expanded.copy()
            set_init = set(map(tuple, connectable_lct_pairs))

            if constellation == "kuiper":
                varying_sg_simulation = kuiper_simulation(ts, sat_array)
            else:
                varying_sg_simulation = Time_Varying_Simulation(ts, sat_array)
            if constellation == "oneweb" or constellation == "kuiper":
                varying_sg_simulation.terrain.GW_RANGE = 1800.0  # in kilometers, range of the ground station
                varying_sg_simulation.LISL_MAX_DISTANCE = 4000.0  # in kilometers, max distance for laser links  
            varying_sg_simulation.reset_simulation_time()
            varying_sg_simulation.config_l_mask(seed=seed)
            varying_sg_simulation.update_space()

            
            upper_time_us = 1e8
            lower_time_us = 1e3
            stop_delta_us = 1e3
            
            while upper_time_us - lower_time_us > stop_delta_us:
                mid_time_us = (upper_time_us + lower_time_us) / 2
                print(f"Running simulation with step: {mid_time_us}")
                varying_sg_simulation.reset_simulation_time()
                varying_sg_simulation.step_time_us(mid_time_us)
                varying_sg_simulation.update_space()
                tmp_solver = ldl_sg_compare_solver()
                varying_sg_simulation.set_solver(tmp_solver)
                tmp_solver.price_graph.price_graph = sg_solver.price_graph.price_graph.copy()
                varying_sg_simulation.update_solver_traffic_info(seed=seed)
                
                now_connectable_lct_pairs = varying_sg_simulation.filtered_lct_pair_expanded.copy()
                
                # Compare the connectable LCT pairs (k,2) array
                set_now = set(map(tuple, now_connectable_lct_pairs))
                
                removed_links = set_init - set_now
                
                # number of removed links
                num_removed = len(removed_links)
                total_links = len(set_init)                
            
                if num_removed > 0.01 * total_links:
                    upper_time_us = mid_time_us
                else:
                    lower_time_us = mid_time_us
            LOG_CSV_WRITTER.log_mul_scalar("rm_links", 0, [mid_time_us, num_removed, total_links], g_step=seed)
