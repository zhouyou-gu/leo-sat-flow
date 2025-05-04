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

np.set_printoptions(precision=4, suppress=True)

class testsolver(mr_solver):
    def update_step_rates_prices(self):
        self.N_STEP += 1
        step_size = self.ALPHA / (self.N_STEP ** 0.2)
        self._print("Updating edge prices")

        # qx_csr = sp.csr_matrix((self.n_sat, self.n_sat))
        
        prices = self.price_graph.get_prices(self.possible_sat_pair_expanded)
        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, prices)
        self._print(f"do routing: max: {np.max(prices)}, min: {np.min(prices)}")

        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )
        
        self._print(f"Maximize rate: {srouting.shape}")            
        self.s_t_data_rate -= step_size * (-1 + costs)
        self.s_t_data_rate = np.clip(self.s_t_data_rate, 0, None)
        
        self._printalltime(f"Appr rate: MAX: {np.max(self.s_t_data_rate)}, MIN: {np.min(self.s_t_data_rate)}")
        self._printalltime(f"Cost path: MAX: {np.max(costs)}, MIN: {np.min(costs)}")
        
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, self.s_t_data_rate, lengths, paths_all
        )
        self._print(f"Compute qx: {srouting.shape}")
        if np.asarray(srouting).size == 0:
            qx_csr = sp.csr_matrix((self.n_sat, self.n_sat))
        else:
            traffic = srouting[:,4]
            self._printalltime(f"Tr: max: {np.max(traffic)}, min: {np.min(traffic)}")
            qx = build_csr(self.n_sat, srouting[:,0:2], traffic, merging_method='sum')
            self._printalltime(f"qx: max: {np.max(qx[2])}, min: {np.min(qx[2])}")
            qx_csr = sp.csr_matrix((qx[2], qx[1], qx[0]), shape=(self.n_sat, self.n_sat))

        connected_sat, connected_lct = self.get_dual_matching()
        self._print(f"Compute rc: {connected_sat.shape}")
        if np.asarray(connected_sat).size == 0:
            rc_csr = sp.csr_matrix((self.n_sat, self.n_sat))
        else:
            capacity_matched = self.compute_capacity(
                np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
            )
            self._print(f"Cm: max: {np.max(capacity_matched)}, min: {np.min(capacity_matched)}")
            rc = build_csr(self.n_sat, connected_sat, capacity_matched, merging_method='sum')
            self._printalltime(f"rc: max: {np.max(rc[2])}, min: {np.min(rc[2])}")
            rc_csr = sp.csr_matrix((rc[2], rc[1], rc[0]), shape=(self.n_sat, self.n_sat))
        
        # Update edge prices
        old_price = self.price_graph.get_prices(self.possible_sat_pair_expanded)
        self._printalltime(f"Oe: max: {np.max(old_price)}, min: {np.min(old_price)}")
        
        self._printalltime(f"diff: {(qx_csr>rc_csr).mean()}")
        
        dif_price_graph = step_size * (qx_csr - rc_csr)
        
        self.price_graph.add_prices(dif_price_graph)     
        new_price = self.price_graph.get_prices(self.possible_sat_pair_expanded)
        self._printalltime(f"Ne: max: {np.max(new_price)}, min: {np.min(new_price)}")
        
        self._print(f"Sz: {step_size}")        
        return new_price    

    # def get_prim_objective(self, with_rates=False):
    #     self._printalltime("Computing prim objective SPF first")
    #     prices = self.price_graph.get_prices(self.possible_sat_pair_expanded)
    #     indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, prices)
    #     costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
    #     srouting = construct_edges_matrix_all_in_one(
    #         self.data_source, self.data_target, self.s_t_data_rate, lengths, paths_all
    #     )
    #     indptr, indices, data = build_csr(self.n_sat, srouting[:,0:2], srouting[:,4], merging_method='sum')
    #     traffic_on_graph_idx_wgt = csr_to_edge_list(indptr, indices, data)
    #     traffic_on_graph_idx_wgt = extract_weights(self.possible_sat_pair_expanded, traffic_on_graph_idx_wgt)
    #     capacity = self.compute_capacity(
    #         np.linalg.norm(self.positions[self.possible_sat_pair_expanded[:, 0]] - self.positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
    #     )
    #     weights  = (capacity + 1e-2)
        
    #     weighted_edges = np.column_stack((self.possible_lct_pair_expanded, weights))
    #     matching = greedy_max_weight_matching(weighted_edges)
    #     m = len(matching)
    #     flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
    #                               for x in pair), dtype=int, count=2*m)
    #     connected_lct = flat_array.reshape(-1, 2)
    #     connected_sat = connected_lct // self.N_LCT_PER_SAT
        
        
    #     self._printalltime(f"Redo routing")
    #     prices = self.price_graph.get_prices(connected_sat)
    #     indptr, indices, data = build_csr(self.n_sat, connected_sat, prices)
    #     costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        
    #     srouting = construct_edges_matrix_all_in_one(
    #         self.data_source, self.data_target, costs, lengths, paths_all
    #     )
                
    #     rates = self.get_max_rate(srouting)
    #     self._printalltime(f"Real rate: MAX: {np.max(rates)}, MIN: {np.min(rates)}")
    #     if not with_rates:
    #         return -np.sum(rates)
    #     else:
    #         return -np.sum(rates), rates, srouting



class DualSimulation(Simulation):
    def set_solver(self, solver):
        self.solver:mr_solver = solver
        self.solver.init_constellation(self.filtered_repeated, self.filtered_expanded, self.positions)

    def run_step(self):
        # Check the constellation connectivity
        connected, comp = self.solver.check_connected()
        self.solver.update_step_rates_prices()
         
        # Evaluate g(lambda)
        d_o, edge_prices = self.solver.get_dual_objective(with_prices=True)
        # Compute the prim p.
        p_o, rates, srouting = self.solver.get_prim_objective(with_rates=True)
        
        gap = p_o - d_o
        # Compute the gap.
        print(f"Gap: {gap:.3f}, Exp: {np.exp(-(gap)):.3f}, d_o: {d_o:.3f}, p_o: {p_o:.3f}")
        avg_p_o = self._moving_average("p_o", p_o)
        print(f"avg: {avg_p_o}")
        self._add_np_log("gap", self.N_STEP, np.array([gap]))
        self._add_np_log("p_o", self.N_STEP, np.array([p_o]))
        
        # self.srouting = self.solver.get_dual_srouting()
        # logger.debug(f"Routing shape: {self.srouting.shape}")
        self._update_o_lisl(satp=srouting[:,0:2].astype(np.int64), edge_weight=None, viz=self.viz_list[0]) 

        vis_prices = edge_prices[:,2]/np.max(edge_prices[:,2])
        self._update_o_lisl(satp=edge_prices[:,0:2].astype(np.int64), edge_weight=vis_prices, viz=self.viz_list[1]) 
        
        # # Update the step edge prices.
        # self.solver.update_step_edge_prices(self.connected_lct, self.srouting)
        # prices = self.solver.price_graph.get_prices(self.filtered_repeated)
        # edge_weight = prices/np.max(prices)
        # self._update_o_lisl(satp=self.filtered_repeated, edge_weight=edge_weight, viz=self.viz_list[2])
    
        # # Compute the prim srouting.
        # traffic_load_and_capacity = self.solver.get_prim_srouting()
        # overflow = traffic_load_and_capacity[:,2] > traffic_load_and_capacity[:,3]
        # overflow = overflow.astype(np.float32).flatten()
        # self._update_o_lisl(satp=traffic_load_and_capacity[:,0:2].astype(np.int64), edge_weight=overflow, viz=self.viz_list[3], binary=True)

# Load tle data.
# ts, valid_satellites, sat_array = generate_walker_constellation_add_planes(planes=20)
ts, valid_satellites, sat_array = generate_walker_constellation(planes=20)

# Create simulation instance.
simulation = DualSimulation(ts, sat_array)

solver = testsolver()
solver.INIT_PRICES = 0.

# solver._debug()
simulation.set_solver(solver)

solver.update_source_target_pairs(20,data_rate=0,seed=0)

simulation.run(N_STEPS=1000,visualize=True)

from sim_src.util import plot_a_array, LOGGED_NP_DATA_HEADER_SIZE
import os
path = os.path.join(os.path.dirname(__file__))

gap = simulation.LOGGED_NP_DATA["gap"][:,LOGGED_NP_DATA_HEADER_SIZE]
gap = gap[gap < np.inf]

title = f"Init price {solver.INIT_PRICES}"
if np.asarray(gap).size != 0:
    plot_a_array(gap, name="gap", title=title, save_path=path)
else:
    plot_a_array(np.zeros_like((simulation.LOGGED_NP_DATA["p_o"][:,LOGGED_NP_DATA_HEADER_SIZE])), name="gap", title=title, save_path=path)
plot_a_array(simulation.LOGGED_NP_DATA["p_o"][:,LOGGED_NP_DATA_HEADER_SIZE], name="p_o", title=title, save_path=path)