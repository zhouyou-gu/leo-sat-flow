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

from sim_mld.ml.sgl.model import lpd_model
from sim_mld.solver import *
from sim_mld.visual import *
from sim_mld.tle import *
from sim_mld.constellation import *
from sim_mld.simulation import Simulation

from vispy import app
import logging

from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT, STATS_OBJECT, counted
from working_dir_path import get_working_dir_path

import torch
torch.set_float32_matmul_precision('medium')

np.set_printoptions(precision=4, suppress=True)

class gnnsolver(mr_solver):
    def init_gnn(self, path=None):
        self.model = lpd_model()
        LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
    
    def update_step_rates_prices(self):
        self.N_STEP += 1
        capacity = self.compute_capacity(
            np.linalg.norm(self.positions[self.possible_sat_pair_expanded[:, 0]] - self.positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
        )
        indptr, indices, capacity = build_csr(self.n_sat, self.possible_sat_pair_expanded, capacity, merging_method='sum')
        cp_edge_list = csr_to_edge_list(indptr, indices, capacity)
        possible_sat_pair_non_expanded = cp_edge_list[:, 0:2]
        possible_capacity_non_expanded = cp_edge_list[:, 2]
        st_edge_index = np.column_stack((self.data_source, self.data_target))
        
        qrates, prices = self.model.get_output_np_edge_weight(self.positions, possible_sat_pair_non_expanded, possible_capacity_non_expanded, st_edge_index, use_target=True)
        
        self._printalltime(f"qrates: max: {np.max(qrates)}, min: {np.min(qrates)}, avg: {np.mean(qrates)}")
        self._printalltime(f"prices: max: {np.max(prices)}, min: {np.min(prices)}, avg: {np.mean(prices)}")
        
        print("shapes", qrates.shape, prices.shape)
        edge_weights_pr = prices * capacity
        
        edge_weights_pr_list = np.column_stack((possible_sat_pair_non_expanded, edge_weights_pr))
        edge_weights_pr = extract_weights(self.possible_sat_pair_expanded, edge_weights_pr_list)
        
        weighted_edges = np.column_stack((self.possible_lct_pair_expanded, edge_weights_pr))
        matching = greedy_max_weight_matching(weighted_edges)
        m = len(matching)
        flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
                                  for x in pair), dtype=int, count=2*m)
        connected_lct = flat_array.reshape(-1, 2)
        connected_sat = connected_lct // self.N_LCT_PER_SAT
        
        
        prices_edge_list = np.column_stack((possible_sat_pair_non_expanded, prices))
        prices = extract_weights(self.possible_sat_pair_expanded, prices_edge_list)
        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, prices)
        self._printalltime(f"prices: max: {np.max(prices)}, min: {np.min(prices)}")

        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        
        valid_costs = costs[lengths > 0]
        self._printalltime(f"costs : max: {np.max(valid_costs)}, min: {np.min(valid_costs)}, avg: {np.mean(valid_costs)}")

        connected_st = lengths > 0
                
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, qrates, lengths, paths_all
        )
         
        if np.asarray(srouting).size == 0 or np.asarray(connected_sat).size == 0:
            return
        
        qx = build_csr(self.n_sat, srouting[:,0:2], srouting[:,4], merging_method='sum')
        qx_edge_list = csr_to_edge_list(qx[0], qx[1], qx[2])
        qx_weights = extract_weights(possible_sat_pair_non_expanded, qx_edge_list)
        qx_edge_list = np.column_stack((possible_sat_pair_non_expanded, qx_weights))
        
        capacity_matched = self.compute_capacity(
            np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
        )
        rc = build_csr(self.n_sat, connected_sat, capacity_matched, merging_method='sum')
        rc_edge_list = csr_to_edge_list(rc[0], rc[1], rc[2])
        rc_weights = extract_weights(possible_sat_pair_non_expanded, rc_edge_list)
        rc_edge_list = np.column_stack((possible_sat_pair_non_expanded, rc_weights))
        
        
        data = {}
        data["x"] = self.positions
        data["cp_edge_index"] = cp_edge_list[:, 0:2]
        data["cp_edge_attr"] = cp_edge_list[:, 2]
        data["qx_edge_index"] = qx_edge_list[:, 0:2]
        data["qx_edge_attr"] = qx_edge_list[:, 2]
        data["rc_edge_index"] = rc_edge_list[:, 0:2]
        data["rc_edge_attr"] = rc_edge_list[:, 2]
        data["st_edge_index"] = st_edge_index[connected_st]
        data["st_edge_attr"] = costs[connected_st]

        self.model.step(data)

        _, new_prices = self.model.get_output_np_edge_weight(self.positions, possible_sat_pair_non_expanded, possible_capacity_non_expanded, st_edge_index, use_target=True)

        self.price_graph.price_graph = sp.csr_matrix((new_prices, (possible_sat_pair_non_expanded[:, 0], possible_sat_pair_non_expanded[:, 1])), shape=(self.n_sat, self.n_sat))
        return

    def get_prim_objective(self, with_rates=False):
        self._print("Computing prim objective")
        connected_sat, connected_lct = self.get_dual_matching()

        prices = self.price_graph.get_prices(connected_sat)
        indptr, indices, data = build_csr(self.n_sat, connected_sat, prices)
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )
        
        rates = self.get_rates(srouting,connected_lct,mode=self.objective_mode)
        self._print(f"Real rate: MAX: {np.max(rates)}, MIN: {np.min(rates)}")
        self._print(f"Appr rate: MAX: {np.max(self.s_t_data_rate)}, MIN: {np.min(self.s_t_data_rate)}")
        if not with_rates:
            return -np.sum(rates)
        else:
            return -np.sum(rates), rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct


class DualSimulation(Simulation):
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

    @counted
    def run_step(self):
        if self.filtered_expanded.size == 0:
            return
        # Check the constellation connectivity
        self.solver.update_step_rates_prices()
        if self.N_STEP % 20 == 0:
            p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = self.solver.get_prim_objective(with_rates=True)
            p_o_mwm, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = self.solver.get_prim_objective_mwm(with_rates=True)
            self._printalltime(f"Prim objective: {p_o}, MWM: {p_o_mwm}")
            self._add_np_log("objective", self.N_STEP, [p_o, p_o_mwm])

    def run(self, N_STEPS=1000, visualize=False):
        """
        Run the simulation.
        """
        for i in range(N_STEPS):
            self.config_l_mask(lct2_rho=rho, lct4_rho=rho, seed=i)
            self.update_space()
            self._init_solver()
            self.solver.update_source_target_pairs(1000, seed=i)
            self.update(None)

        print("Simulation completed.")


# Load tle data.
ts, valid_satellites, sat_array = generate_walker_constellation(planes=20)

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

# Create simulation instance.
rho = 0.3
print(f"Running simulation with rho={rho}")
simulation = DualSimulation(ts, sat_array)
simulation.update_space()

solver = gnnsolver()
solver.init_gnn()

simulation.set_solver(solver)

simulation.run(N_STEPS=5000)

simulation.save_np(LOG_DIR,"final")