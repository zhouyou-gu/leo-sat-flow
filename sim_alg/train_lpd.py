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

from sim_mld.ml.ld_sgl.model import ld_model
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
        self.model = ld_model()
    
    @counted
    def update_step_rates_prices(self):
        capacity = self.compute_capacity(
            np.linalg.norm(self.positions[self.possible_sat_pair_expanded[:, 0]] - self.positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
        )
        indptr, indices, capacity = build_csr(self.n_sat, self.possible_sat_pair_expanded, capacity, merging_method='sum')
        cp_edge_list = csr_to_edge_list(indptr, indices, capacity)
        print(f"cp_edge_list shape: {cp_edge_list.shape}, possible_sat_pair_expanded shape: {self.possible_sat_pair_expanded.shape}")
        possible_sat_pair_non_expanded = cp_edge_list[:, 0:2]
        possible_capacity_non_expanded = cp_edge_list[:, 2]

        sat_capacity = self.forward_traffic_capacity/self.forward_traffic_capacity.mean()
        sat_demand = self.forward_traffic_demand/self.forward_traffic_demand.mean()
        self._printalltime(f"sat_capacity: max: {np.max(sat_capacity)}, min: {np.min(sat_capacity)}, avg: {np.mean(sat_capacity)}, shape: {sat_capacity.shape}")
        self._printalltime(f"sat_demand: max: {np.max(sat_demand)}, min: {np.min(sat_demand)}, avg: {np.mean(sat_demand)}, shape: {sat_demand.shape}")
        x = np.concatenate((sat_capacity.reshape(-1, 1), sat_demand.reshape(-1, 1)), axis=1)
        prices = self.model.get_output_np_edge_weight(x,possible_sat_pair_non_expanded, possible_capacity_non_expanded, use_target=False)

        self._printalltime(f"prices: max: {np.max(prices)}, min: {np.min(prices)}, avg: {np.mean(prices)}")
        self._printalltime(f"shapes of prices: {prices.shape}")
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
        self._printalltime(f"data_source: {self.data_source}, data_target: {self.data_target}")
        self._printalltime(f"shapes of data_source, data_target: {self.data_source.shape}, {self.data_target.shape}")
        tic = self._get_tic()
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        tim = self._get_tim(tic)
        self._printalltime(f"multi_dijkstra_with_paths time: {tim:.4f} us")
        valid_costs = costs[lengths > 0]
        self._printalltime(f"costs : max: {np.max(valid_costs)}, min: {np.min(valid_costs)}, avg: {np.mean(valid_costs)}")

        connected_st = lengths > 0
        self._printalltime(f"connected_st: {np.sum(connected_st)} out of {self.data_source.shape[0]} pairs")
        
        self._printalltime(f"starting to compute rates")
        tic = self._get_tic()
        self.s_t_traffic_rates = self.get_rates_dual(costs=costs)
        print(f"rates shape: {self.s_t_traffic_rates.shape}, costs shape: {costs.shape}")
        print("self.s_t_traffic_rates[costs>1].mean():", self.s_t_traffic_rates[costs>1].mean())
        print(f"approximate rates: max: {np.max(self.s_t_traffic_rates)}, min: {np.min(self.s_t_traffic_rates)}, avg: {np.mean(self.s_t_traffic_rates)}")
        tim = self._get_tim(tic)
        self._printalltime(f"Computed rates: {self.s_t_traffic_rates.shape}, Time: {tim:.4f} us")

        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, self.s_t_traffic_rates, lengths, paths_all
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
        data["x"] = x
        data["cp_edge_index"] = cp_edge_list[:, 0:2]
        data["cp_edge_attr"] = cp_edge_list[:, 2]
        data["qx_edge_index"] = qx_edge_list[:, 0:2]
        data["qx_edge_attr"] = qx_edge_list[:, 2]
        data["rc_edge_index"] = rc_edge_list[:, 0:2]
        data["rc_edge_attr"] = rc_edge_list[:, 2]

        self.model.step(data)

        new_prices = self.model.get_output_np_edge_weight(x, possible_sat_pair_non_expanded, possible_capacity_non_expanded)

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

        rates = self.get_rates_prim(srouting, connected_lct, mode=self.objective_mode)
        self._print(f"Real rate: MAX: {np.max(rates)}, MIN: {np.min(rates)}")
        self._print(f"Appr rate: MAX: {np.max(self.s_t_traffic_rates)}, MIN: {np.min(self.s_t_traffic_rates)}")
        if not with_rates:
            return -np.sum(rates)
        else:
            return -np.sum(rates), rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct


class DualSimulation(Simulation):
    def config_l_mask(self, lct2_rho=0.1, lct4_rho=0.2, seed=0):
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

    def get_simulation_time(self):
        # return time at 2025 jun 1st
        return self.ts.utc(2025, 6, 1, 0, 0, 0)
    
    @counted
    def run_step(self):
        self.config_l_mask(seed=self.N_STEP)
        self.update_space()
        self.update_solver_constellation_info()
        self.update_solver_traffic_info(seed=self.N_STEP)
        
        if self.filtered_expanded.size == 0:
            return
        
        # Check the constellation connectivity
        connected, comp = self.solver.check_connected()
        self._printalltime(f"Connected: {connected}, Components: {comp}")
        
        self.solver.update_step_rates_prices()
         
        tic = self._get_tic()
        p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = self.solver.get_prim_objective(with_rates=True)
        toc = self._get_tim(tic)
        self._printalltime(f"Prim objective: {p_o}, Time: {toc:.4f} us")
        p_o_mwm, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = self.solver.get_prim_objective_mwm(with_rates=True)
        self._printalltime(f"Prim objective: {p_o}, MWM: {p_o_mwm}, Ratio: {p_o/p_o_mwm}")
        self._add_np_log("objective", self.N_STEP, [p_o, p_o_mwm])

    def run(self, TOT_STEPS=1000, visualize=False):
        for i in range(TOT_STEPS):
            self.run_step()
            print(f"++++++++++++++++Step {i+1}/{TOT_STEPS} completed++++++++++++++++")
        return 

# Load tle data.
ts, valid_satellites, sat_array = generate_walker_constellation(planes=20)

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

# Create simulation instance.
simulation = DualSimulation(ts, sat_array)
simulation.update_space()

solver = gnnsolver()
solver.init_gnn()

simulation.set_solver(solver)
simulation.run(TOT_STEPS=5000)

simulation.save_np(LOG_DIR,"final")