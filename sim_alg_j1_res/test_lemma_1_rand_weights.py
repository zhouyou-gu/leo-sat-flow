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

class lemma1(mr_solver):
    @counted
    def update_step_rates_prices(self):
        print("Updating step rates and prices")
        capacity = self.compute_capacity(
            np.linalg.norm(self.positions[self.possible_sat_pair_expanded[:, 0]] - self.positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
        )

        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, capacity, merging_method='avg', sym_half=True)
        

        rng = np.random.default_rng(seed=step)

        data = rng.lognormal(size=data.size, mean=0., sigma=1.)


        self.price_graph.price_graph = sp.csr_matrix((data, indices, indptr), shape=(self.n_sat, self.n_sat))

        plotext.title("Random Price Distribution")
        plotext.hist(np.log10(self.price_graph.price_graph.data + 1e-5), bins=50, norm=True)
        plotext.plotsize(100, 30)
        plotext.show()
        plotext.clf()
        return
    
    def clip_prices(self):
        self.price_graph.price_graph.data = np.clip(self.price_graph.price_graph.data, 0, 1.0)
        plotext.title("Clip Price Distribution")
        plotext.hist(np.log10(self.price_graph.price_graph.data + 1e-5), bins=50, norm=True)
        plotext.plotsize(100, 30)
        plotext.show()
        plotext.clf()
        return

class Lemma1Simulation(Simulation):
    def get_simulation_time(self):
        # return time at 2025 jun 1st
        return self.ts.utc(2025, 7, 16, 16, 0, 0)

    
    def run_step(self):
        self._printalltime(f"run_step")

        if self.filtered_lct_pair_expanded.size == 0:
            return
        
        # Check the constellation connectivity
        connected, comp = self.solver.check_connected()
        self._printalltime(f"Connected: {connected}")
        self._printalltime("Updating solver traffic info")
        self.update_solver_traffic_info(seed=self.N_STEP)
        
        self.solver.update_step_rates_prices()
         
        # Evaluate g(lambda)
        d_o_1, o_1_rate_cost, o_1_rate_mtch = self.solver.get_dual_objective(with_objective=True)
        
        self.solver.clip_prices()

        d_o_2, o_2_rate_cost, o_2_rate_mtch = self.solver.get_dual_objective(with_objective=True)

        
        # Compute the prim p.
        self._add_np_log("dual_objective", self.N_STEP, [o_1_rate_cost, o_1_rate_mtch, o_2_rate_cost, o_2_rate_mtch])
        self._printalltime(f"Dual Objective: {d_o_1:.4f}:{d_o_2:.4f}, Rate Cost: {o_1_rate_cost:.4f}:{o_2_rate_cost:.4f}, Rate Mtch: {o_1_rate_mtch:.4f}:{o_2_rate_mtch:.4f}")
        return o_1_rate_cost, o_1_rate_mtch, o_2_rate_cost, o_2_rate_mtch
        
# Load tle data.
from working_dir_path import get_working_dir_path
import os
tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)


solver = lemma1()
for step in range(500):
    print(f"Running simulation with step: {step}")
    # Create simulation instance.
    ts, valid_satellites, sat_array = generate_tle_partly_regular_constellation1000(n_sat=1000, ratio=0.0, starlink_tle_path=tle_file_path, seed=step)
    simulation = Lemma1Simulation(ts, sat_array)
    simulation.config_l_mask(seed=step)

    simulation.update_space()
    simulation.set_solver(solver)
    o_1_rate_cost, o_1_rate_mtch, o_2_rate_cost, o_2_rate_mtch = simulation.run_step()
    LOG_CSV_WRITTER.log_mul_scalar("d_o_component", step, [o_1_rate_cost, o_1_rate_mtch, o_2_rate_cost, o_2_rate_mtch], g_step=step)


