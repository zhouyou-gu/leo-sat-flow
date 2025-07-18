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

class sdfsolver(mr_solver):    
    def get_prim_objective(self, with_rates=False):
        self._print("Computing get_prim_objective_srouting_first")
        distance = np.linalg.norm(self.positions[self.possible_sat_pair_expanded[:, 0]] - self.positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
        capacity = self.compute_capacity(distance)
        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, (1./capacity)**(1.3))
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )
        
        indptr, indices, path_selected = build_csr(self.n_sat, srouting[:, 0:2], np.ones_like(srouting[:,4]), merging_method="sum")

        edge_list_path_selected = csr_to_edge_list(indptr, indices, path_selected)
        weights = extract_weights(self.possible_sat_pair_expanded, edge_list_path_selected)
        weighted_edges = np.column_stack((self.possible_lct_pair_expanded, weights))
        matching = greedy_max_weight_matching(weighted_edges)
        m = len(matching)
        flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
                                  for x in pair), dtype=int, count=2*m)
        connected_lct = flat_array.reshape(-1, 2)
        connected_sat = connected_lct // self.N_LCT_PER_SAT

        distance = np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
        capacity = self.compute_capacity(distance)
        indptr, indices, data = build_csr(self.n_sat, connected_sat, (1./capacity)**(1.3))
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )
        
        rates = self.get_rates(srouting, connected_lct, mode=self.objective_mode)
        self._print(f"Real rate: MAX: {np.max(rates)}, MIN: {np.min(rates)}")
        if not with_rates:
            return -np.sum(rates)
        else:
            return -np.sum(rates), rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct
      

class DualSimulation(Simulation):
    def set_solver(self, solver):
        self.solver:mr_solver = solver
        self.solver.init_constellation(self.filtered_sat_pair_repeated, self.filtered_lct_pair_expanded, self.positions)

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
        
        
# Load tle data.
ts, valid_satellites, sat_array = generate_walker_constellation(planes=20)

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

# Create simulation instance.
rho = 0.3
i = 0
print(f"Running simulation with rho={rho}, SEED={i}")
simulation = DualSimulation(ts, sat_array)
simulation.config_l_mask(lct2_rho=rho,lct4_rho=rho, seed=i)
simulation.update_space()

solver = sdfsolver()
solver.INIT_PRICES = 1.

simulation.set_solver(solver)

solver.update_source_target_pairs(1000,data_rate=0,seed=i)

p_o, rates, srouting, srouting_tuple, connected_sat, connected_lct = solver.get_prim_objective(with_rates=True)

simulation._update_o_lisl(satp=connected_sat, viz=simulation.viz_list[1])


print(f"p_o: {p_o:.3f}, rates: {rates.mean():.3f}， rates max: {rates.max():.3f}, rates min: {rates.min():.3f}")
LOG_OBJ._add_np_log("p_o", i, np.array([p_o]))
print(f"log_rates: {np.log(rates+0.1).mean():.3f}， log_rates max: {np.log(rates+1).max():.3f}, log_rates min: {np.log(rates+1).min():.3f}")       
print(f"null_count: {np.sum(rates == 0)}")
from sim_src.util import plot_a_array, LOGGED_NP_DATA_HEADER_SIZE
import os
path = os.path.join(os.path.dirname(__file__))

rates.sort()
plot_a_array(rates, mavg_n=None,name="rate-sdf", title="sdf", save_path=path)
app.run()