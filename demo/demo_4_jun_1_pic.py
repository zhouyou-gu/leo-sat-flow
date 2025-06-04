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

from sim_mld.solver import *
from sim_mld.visual import *
from sim_mld.tle import *
from sim_mld.constellation import *
from sim_mld.simulation import Simulation

from vispy import app
import logging

from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT, STATS_OBJECT

np.set_printoptions(precision=4, suppress=True)

class Demo(Simulation):
    def config_l_mask(self, lct2_rho=0., lct4_rho=0., seed=0):
        assert lct2_rho + lct4_rho <= 1, "lct2_rho + lct4_rho must be less than or equal to 1"
        n_lct2_sat = int(self.n_sat * lct2_rho)
        n_lct4_sat = int(self.n_sat * lct4_rho)
        
        rng = np.random.default_rng(seed)
        
        permuted_indices = rng.permutation(np.arange(0, self.n_sat))
        self.lct2_indices = permuted_indices[:n_lct2_sat]
        self.lct4_indices = permuted_indices[n_lct2_sat:n_lct2_sat + n_lct4_sat]
        
        self.lct_mask = np.zeros((self.n_sat, self.N_LCT_PER_SAT), dtype=np.float32)
        self.lct_mask[:,1] = 1
        self.lct_mask[self.lct2_indices] = np.array([1, 1, 0, 0], dtype=np.float32)
        self.lct_mask[self.lct4_indices] = np.array([1, 1, 1, 1], dtype=np.float32)   

    def update_solver_traffic_info(self, seed=0):
        self.solver.data_source, self.solver.data_target, self.solver.forward_traffic_capacity, self.solver.forward_traffic_demand = self.terrain.get_traffic_info_test(self.positions,seed=seed)
        self.solver.s_t_traffic_rates = np.zeros(self.solver.data_source.shape[0], dtype=np.float32)
        self._printalltime(f"Before s_t pair filtering seed {seed}, source: {self.solver.data_source.shape[0]}, target: {self.solver.data_target.shape[0]}")
        self.solver._remove_non_connected_s_t_pairs()
        self._printalltime(f"Updated traffic info with seed {seed}, source: {self.solver.data_source.shape[0]}, target: {self.solver.data_target.shape[0]}")


    def run_step(self):
        time.sleep(0.01)  # Simulate some processing time
        data_source, data_target = self.solver.data_source, self.solver.data_target
        
        satp = np.stack((data_source, data_target), axis=1)

        self._update_traffic_flow(satp=satp, edge_weight=None, viz=self.viz_list[3])
        p_o, rates, srouting, srouting_tuple, connected_sat, connected_lct = self.solver.get_prim_objective_mwm(with_rates= True)
        
        
        capacity = self.solver.compute_capacity(np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1))
        
        capacity = capacity/ capacity.max()  # Normalize capacity for visualization
        capacity = np.log1p(capacity*20)  # Apply log1p for better visualization
        capacity = capacity/ capacity.max()  # Normalize again after log1p
        self._update_o_lisl(satp=connected_sat, viz=self.viz_list[2], edge_weight=capacity/capacity.max())
        
        

    def _update_traffic_flow(self, satp, edge_weight=None, viz=None):
        if viz is None:
            return
        
        if "traffic_flow" in viz:
            edge_from = self.positions[satp[:, 0]]
            edge_to = self.positions[satp[:, 1]]
            if edge_weight is None:
                edge_weight = np.ones(satp.shape[0], dtype=self.EDGE_COLOR.dtype)
            
            COLOR_FROM_BASE = 0.25
            color_data_to = np.zeros((edge_weight.shape[0], 4), dtype=self.EDGE_COLOR.dtype)
            edge_weight = np.tanh(edge_weight).clip(0, 1)*(1-COLOR_FROM_BASE) +COLOR_FROM_BASE
            color_data_to[:, 3] = edge_weight
            color_data_from = np.zeros((edge_weight.shape[0], 4), dtype=self.EDGE_COLOR.dtype)
            color_data_from[:, 3] = COLOR_FROM_BASE
            color_data = np.concatenate((color_data_from, color_data_to), axis=1).reshape(-1, 4)
            pos_data = np.concatenate((edge_from, edge_to), axis=1).reshape(-1, 3)
            
            # arrow_dir = np.concatenate((edge_from, edge_to), axis=1)
            
            viz['traffic_flow'].set_data(pos=pos_data, color=color_data,
                                         width=0.1, connect='segments')

    def set_viz_data(self):
        self.viz_list[0]['text_top_left'].text = f"Step {self.N_STEP}/{self.TOT_STEPS}"
        self.viz_list[0]['text_bot_left'].text = f"Time {time.perf_counter()-self.real_start_time:.2f}s"


    def setup_visualization(self):
        self.canvas, self.viz_list = setup_viz_list_one_canvas(sceen_size=(1200, 800), shape=(2, 2))
        self.viz_list[3]['sphere_visual'].visible = False
        traffic_flow = scene.visuals.Arrow()
        self.viz_list[3]['view'].add(traffic_flow)
        self.viz_list[3]['traffic_flow'] = traffic_flow       
        self.viz_list[0]['text_title'].text = f"Constellation" 
        self.viz_list[1]['text_title'].text = f"Earth Terrain: Population, and Ground Stations" 
        self.viz_list[2]['text_title'].text = f"Laser Inter-Satellite Links"
        self.viz_list[3]['text_title'].text = f"Traffic Source-Destination Pairs"
    
        self.viz_list[1]['scatter'].visible = False
        self.viz_list[1]['arrow'].visible = False
        self.viz_list[2]['scatter'].visible = False
        self.viz_list[2]['o_scatter'].visible = False
        self.viz_list[2]['sphere_visual'].visible = False
        for viz in self.viz_list:
            viz['axes'].visible = False
            viz['view'].camera.azimuth = self.compute_rotation() + 90
            viz['view'].camera.elevation = 30
            viz['view'].camera.distance = 3

# Load tle data.
ts, valid_satellites, sat_array = generate_walker_constellation(planes=20)

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

# Create simulation instance.
rho2 = 0.3
rho4 = 0.3
i = 0
print(f"Running simulation with lct2_rho={rho2}, lct4_rho={rho4}, seed={i}")
simulation = Demo(ts, sat_array)
simulation.config_l_mask(lct2_rho=rho2, lct4_rho=rho4, seed=i)
simulation.update_space()

solver = mr_solver()
simulation.set_solver(solver)
simulation.update_solver_traffic_info(seed=i)

simulation.run(TOT_STEPS=50000,visualize=True)