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

class TestViewSimulation(Simulation):
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

    def run_step(self):
        if self.filtered_lct_pair_expanded.size == 0:
            return
        # self._update_o_lisl(satp=connected_sat, viz=self.viz_list[1], edge_weight=capacity/capacity.max())

        # self._update_o_lisl(satp=srouting[:,0:2].astype(np.int64), edge_weight=None, viz=self.viz_list[0]) 

        # vis_prices = edge_prices[:,2]/np.max(edge_prices[:,2]+1e-10)
        # self._update_o_lisl(satp=edge_prices[:,0:2].astype(np.int64), edge_weight=vis_prices, viz=self.viz_list[2]) 
    
        # self.viz_list[0]['text_title'].text = f"Weighted Shortest Path Routing\n - Average Hops {srouting_tuple[1].mean():.2f}" 
        # self.viz_list[1]['text_title'].text = f"Maximum Weight Laser Matching\n - Average Per Link Rate (Gbps) {capacity.mean():.2f}" 
        # self.viz_list[2]['text_title'].text = f"Laser Link Pricing (Dual Variables)\n - Average Price (Gbps Per Hop) {edge_prices[:,2].mean():.2f}"
        # self.viz_list[3]['text_title'].text = f"Source-to-Target Traffic Flow\n - Average Rate (Gbps) {dual_rates.mean():.2f}"
        gs = self.terrain.get_ground_station_positions()
        self.terrain.get_traffic_info(self.positions)
        self.viz_list[0]['o_scatter'].set_data(gs, face_color=[0, 0, 0, 0.5], size=10, edge_width_rel=0)

    def run(self, N_STEPS=1000, visualize=False):
        """
        Run the simulation.
        """
        self.update_space()
        self._print("Starting simulation...")
        self.viz_list[3]['sphere_visual'].visible = False
        traffic_flow = scene.visuals.Arrow()
        self.viz_list[3]['view'].add(traffic_flow)
        self.viz_list[3]['traffic_flow'] = traffic_flow
        
        # self.viz_list[1]['text_top_left'].text = "hello"
        for i in range(N_STEPS):
            self.viz_list[0]['text_top_left'].text = f"Step {i+1}/{N_STEPS}"
            self.viz_list[0]['text_bot_left'].text = f"Time {time.perf_counter()-self.real_start_time:.2f}s"
            if visualize:
                self.canvas.update()   # schedule a redraw
                app.process_events()   # keep GUI alive
            self.update(None)
            self.update_space()
            self.update_simulation_time()
        
        print("Simulation completed.")        
        

# Load tle data.
ts, valid_satellites, sat_array = generate_walker_constellation(planes=20)


# Create simulation instance.
rho = 0.3
simulation = TestViewSimulation(ts, sat_array)
simulation.config_l_mask(lct2_rho=rho, lct4_rho=rho, seed=0)
simulation.update_space()


simulation.run(N_STEPS=50000,visualize=True)
