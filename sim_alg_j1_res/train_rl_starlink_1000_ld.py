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

class gnnsolver(mr_solver):
    def init_gnn(self, path=None, BETA = 0.5, GAMMA=1):
        print("Initializing GNN model")
        self.model = ld_model(BETA=BETA, GAMMA=GAMMA)

    @counted
    def update_step_rates_prices(self):
        print("Updating step rates and prices")
        ## get price
        capacity = self.compute_capacity(
            np.linalg.norm(self.positions[self.possible_sat_pair_expanded[:, 0]] - self.positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
        )
        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, capacity, merging_method='avg', sym_half=True)
        cp_edge_list = csr_to_edge_list(indptr, indices, data)
        print(f"cp_edge_list shape: {cp_edge_list.shape}, possible_sat_pair_expanded shape: {self.possible_sat_pair_expanded.shape}")
        possible_sat_pair_non_expanded_sym = cp_edge_list[:, 0:2]
        possible_capacity_non_expanded_sym = cp_edge_list[:, 2]
        
        sat_capacity = self.forward_traffic_capacity
        sat_demand = self.forward_traffic_demand
        self._printalltime(f"sat_capacity: max: {np.max(sat_capacity)}, min: {np.min(sat_capacity)}, avg: {np.mean(sat_capacity)}, shape: {sat_capacity.shape}")
        self._printalltime(f"sat_demand: max: {np.max(sat_demand)}, min: {np.min(sat_demand)}, avg: {np.mean(sat_demand)}, shape: {sat_demand.shape}")
        x = np.concatenate((sat_capacity.reshape(-1, 1), sat_demand.reshape(-1, 1)), axis=1)
        prices = self.model.get_output_np_edge_weight(x,possible_sat_pair_non_expanded_sym, possible_capacity_non_expanded_sym, use_target=False)
        prices_edge_list = np.column_stack((possible_sat_pair_non_expanded_sym, prices))

        ## get weights
        self._printalltime(f"prices: max: {np.max(prices)}, min: {np.min(prices)}, avg: {np.mean(prices)}")
        self._printalltime(f"shapes of prices: {prices.shape}")
        edge_weights_pr = prices * possible_capacity_non_expanded_sym
        
        ## matching
        edge_weights_pr_list = np.column_stack((possible_sat_pair_non_expanded_sym, edge_weights_pr))
        edge_weights_pr = extract_weights(self.possible_sat_pair_expanded, edge_weights_pr_list)
        edge_weights_pr_flip = extract_weights(self.possible_sat_pair_expanded[:, ::-1], edge_weights_pr_list)
        edge_weights_pr = (edge_weights_pr + edge_weights_pr_flip)

        weighted_edges = np.column_stack((self.possible_lct_pair_expanded, edge_weights_pr))
        matching = greedy_max_weight_matching(weighted_edges)
        m = len(matching)
        flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
                                  for x in pair), dtype=int, count=2*m)
        connected_lct = flat_array.reshape(-1, 2)
        connected_sat = connected_lct // self.N_LCT_PER_SAT

        ## get routing
        indptr, indices, data = build_csr(self.n_sat, possible_sat_pair_non_expanded_sym, prices, sym_half = False)
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
        
        # rate allocation
        self._printalltime(f"starting to compute rates")
        tic = self._get_tic()
        self.s_t_traffic_rates = self.get_rates_dual(costs=costs)
        print(f"rates shape: {self.s_t_traffic_rates.shape}, costs shape: {costs.shape}")
        # print("self.s_t_traffic_rates[costs>1].mean():", self.s_t_traffic_rates[costs>1].mean())
        print(f": max: {np.max(self.s_t_traffic_rates)}, min: {np.min(self.s_t_traffic_rates)}, avg: {np.mean(self.s_t_traffic_rates)}")
        tim = self._get_tim(tic)
        self._printalltime(f"Computed rates: {self.s_t_traffic_rates.shape}, Time: {tim:.4f} us")

        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, self.s_t_traffic_rates, lengths, paths_all
        )
         
        if np.asarray(srouting).size == 0 or np.asarray(connected_sat).size == 0:
            return
        
        ## rate per edge
        qx = build_csr(self.n_sat, srouting[:,0:2], srouting[:,4], merging_method='sum', sym_half=False)
        qx_edge_list = csr_to_edge_list(qx[0], qx[1], qx[2])
        qx_weights = extract_weights(possible_sat_pair_non_expanded_sym, qx_edge_list)
        qx_edge_list = np.column_stack((possible_sat_pair_non_expanded_sym, qx_weights))
        
        capacity_matched = self.compute_capacity(
            np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
        )
        rc = build_csr(self.n_sat, connected_sat, capacity_matched, merging_method='sum')
        rc_edge_list = csr_to_edge_list(rc[0], rc[1], rc[2])
        rc_weights = extract_weights(possible_sat_pair_non_expanded_sym, rc_edge_list)
        rc_edge_list = np.column_stack((possible_sat_pair_non_expanded_sym, rc_weights))


        qx_positive = qx_edge_list[:, 2] > 0
        pr_greater1 = prices_edge_list[:, 2]> 1
        if np.any(qx_positive & pr_greater1):
            print("++++++++++++++++Error+++++++++++++++++++++")
            print("qx_edge_attr > 0 and pr_edge_attr > 1")
            print("qx_edge_attr:", qx_edge_list[qx_positive & pr_greater1])
            print("pr_edge_attr:", prices_edge_list[qx_positive & pr_greater1])
            print("++++++++++++++++++++++++++++++++++++++++++")
        
        data = {}
        data["x"] = x
        data["cp_edge_index"] = cp_edge_list[:, 0:2]
        data["cp_edge_attr"] = cp_edge_list[:, 2]
        data["qx_edge_index"] = qx_edge_list[:, 0:2]
        data["qx_edge_attr"] = qx_edge_list[:, 2]
        data["rc_edge_index"] = rc_edge_list[:, 0:2]
        data["rc_edge_attr"] = rc_edge_list[:, 2]
        data["pr_edge_index"] = prices_edge_list[:, 0:2]
        data["pr_edge_attr"] = prices_edge_list[:, 2]
        data["st_pair_index"] = np.column_stack((self.data_source, self.data_target))
        data["st_pair_attr"] = self.s_t_traffic_rates
        
        self.model.step(data)

        new_prices = self.model.get_output_np_edge_weight(x, possible_sat_pair_non_expanded_sym, possible_capacity_non_expanded_sym, use_target=False)

        self.price_graph.price_graph = sp.csr_matrix((new_prices, (possible_sat_pair_non_expanded_sym[:, 0], possible_sat_pair_non_expanded_sym[:, 1])), shape=(self.n_sat, self.n_sat))
        
        d_sym = self.price_graph.price_graph - self.price_graph.price_graph.T
        d_sym.data = np.abs(d_sym.data)
        print(f"d_sym: {d_sym.data}")
        try:
            plotext.title("d_sym Distribution")
            plotext.hist(np.log10(d_sym.data+1e-5), bins=50, norm=True)
            plotext.plotsize(100, 30)
            plotext.show()
            plotext.clf()
        except Exception as e:
            print("Plotext error:", e)
            pass
        return

class GNNSimulation(Simulation):
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
        d_o, edge_prices = self.solver.get_dual_objective(with_prices=True)
        # Compute the prim p.
        tic = self._get_tic()
        p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = self.solver.get_prim_objective(with_rates=True)
        toc = self._get_tim(tic)
        self._printalltime(f"Prim objective: {p_o}, Time: {toc:.4f} us")
        p_o_mwm, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = self.solver.get_prim_objective_heuristic(with_rates=True)
        self._printalltime(f"Prim objective: {p_o}, MWM: {p_o_mwm}, Ratio: {p_o/p_o_mwm}")
        self._add_np_log("objective", self.N_STEP, [p_o, p_o_mwm])
        return p_o/p_o_mwm, p_o, d_o, p_o_mwm