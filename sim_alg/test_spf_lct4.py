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

class spfsolver(mr_solver):
    def update_step_rates_prices(self):
        self.N_STEP += 1
        new_price = self.price_graph.get_prices(self.possible_sat_pair_expanded)
        self._print(f"Ne: max: {np.max(new_price)}, min: {np.min(new_price)}")        
        return new_price
    
    def get_prim_objective(self, with_rates=False):
        self._print("Computing prim objective")
        connected_sat, connected_lct = self.get_dual_matching()
        prices = self.price_graph.get_prices(self.possible_sat_pair_expanded)
        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, prices)
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )
        
        indptr, indices, path_selected = build_csr(self.n_sat, srouting[:, 0:2], np.ones(srouting.shape[0], dtype=np.float32))

        edge_list_path_selected = csr_to_edge_list(indptr, indices, path_selected)
        weights = extract_weights(self.possible_sat_pair_expanded, edge_list_path_selected)
        weighted_edges = np.column_stack((self.possible_lct_pair_expanded, weights))
        matching = greedy_max_weight_matching(weighted_edges)
        m = len(matching)
        flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
                                  for x in pair), dtype=int, count=2*m)
        connected_lct = flat_array.reshape(-1, 2)
        connected_sat = connected_lct // self.N_LCT_PER_SAT


        prices = self.price_graph.get_prices(connected_sat)
        indptr, indices, data = build_csr(self.n_sat, connected_sat, prices)
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )
        
        rates = self.get_max_rate(srouting)
        self._print(f"Real rate: MAX: {np.max(rates)}, MIN: {np.min(rates)}")
        self._print(f"Appr rate: MAX: {np.max(self.s_t_data_rate)}, MIN: {np.min(self.s_t_data_rate)}")
        if not with_rates:
            return -np.sum(rates)
        else:
            return -np.sum(rates), rates, srouting, (costs, lengths, paths_all)
      

class DualSimulation(Simulation):
    def set_solver(self, solver):
        self.solver:mr_solver = solver
        self.solver.init_constellation(self.filtered_repeated, self.filtered_expanded, self.positions)

    def config_l_mask(self, lct2_rho=0., lct4_rho=0., seed=0):
        assert lct2_rho + lct4_rho <= 1, "lct2_rho + lct4_rho must be less than or equal to 1"
        n_lct2_sat = int(self.n_sat * lct2_rho)
        n_lct4_sat = int(self.n_sat * lct4_rho)
        
        rng = np.random.default_rng(seed)
        
        permuted_indices = rng.permutation(np.arange(0, self.n_sat))
        self.lct2_indices = permuted_indices[:n_lct2_sat]
        self.lct4_indices = permuted_indices[n_lct2_sat:n_lct2_sat + n_lct4_sat]
        
        self.lct_mask = np.zeros((self.n_sat, self.N_LCT_PER_SAT), dtype=np.float32)
        self.lct_mask[self.lct2_indices] = np.array([1, 1, 0, 0], dtype=np.float32)
        self.lct_mask[self.lct4_indices] = np.array([1, 1, 1, 1], dtype=np.float32)
        
    def run_step(self):
        if self.filtered_expanded.size == 0:
            return
        # Check the constellation connectivity
        connected, comp = self.solver.check_connected()
        self.solver.update_step_rates_prices()
         
        # Evaluate g(lambda)
        d_o, edge_prices = self.solver.get_dual_objective(with_prices=True)
        # Compute the prim p.
        p_o, rates, srouting, srouting_tuple = self.solver.get_prim_objective(with_rates=True)
        
        s_t = np.column_stack((self.solver.data_source, self.solver.data_target))
        
        gap = p_o - d_o
        # Compute the gap.
        print(f"Gap: {gap:.3f}, Exp: {np.exp(-(gap)):.3f}, d_o: {d_o:.3f}, p_o: {p_o:.3f}")
        avg_p_o = self._moving_average("p_o", p_o)
        print(f"avg: {avg_p_o}")
        self._add_np_log("gap", self.N_STEP, np.array([gap]))
        self._add_np_log("p_o", self.N_STEP, np.array([p_o]))
        
        dual_rates = self.solver.s_t_data_rate
        self._update_traffic_flow(satp=s_t, edge_weight=dual_rates/20, viz=self.viz_list[3])

        connected_sat, connected_lct = self.solver.get_dual_matching()
        capacity = self.solver.compute_capacity(
            np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
        )
        self._update_o_lisl(satp=connected_sat, viz=self.viz_list[1], edge_weight=capacity/capacity.max())

        self._update_o_lisl(satp=srouting[:,0:2].astype(np.int64), edge_weight=None, viz=self.viz_list[0]) 

        vis_prices = edge_prices[:,2]/np.max(edge_prices[:,2]+1e-10)
        self._update_o_lisl(satp=edge_prices[:,0:2].astype(np.int64), edge_weight=vis_prices, viz=self.viz_list[2]) 
    
        self.viz_list[0]['text_title'].text = f"Weighted Shortest Path Routing\n - Average Hops {srouting_tuple[1].mean():.2f}" 
        self.viz_list[1]['text_title'].text = f"Maximum Weight Laser Matching\n - Average Per Link Rate (Gbps) {capacity.mean():.2f}" 
        self.viz_list[2]['text_title'].text = f"Laser Link Pricing (Dual Variables)\n - Average Price (Gbps Per Hop) {edge_prices[:,2].mean():.2f}"
        self.viz_list[3]['text_title'].text = f"Source-to-Target Traffic Flow\n - Average Rate (Gbps) {dual_rates.mean():.2f}"
    
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
            
            arrow_dir = np.concatenate((edge_from, edge_to), axis=1)
            
            viz['traffic_flow'].set_data(pos=pos_data, color=color_data,
                                         width=5, connect='segments', arrows=arrow_dir)

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
        
        print("Simulation completed.")        
        
        
# Load tle data.
ts, valid_satellites, sat_array = generate_walker_constellation(planes=20)

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

# Create simulation instance.
rho_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for rho in rho_list:
    for i in range(1, 10):
        print(f"Running simulation with rho={rho}, SEED={i}")
        simulation = DualSimulation(ts, sat_array)
        simulation.config_l_mask(lct4_rho=rho, seed=i)
        simulation.update_space()

        solver = spfsolver()
        solver.INIT_PRICES = 1.

        simulation.set_solver(solver)

        solver.update_source_target_pairs(20,data_rate=0,seed=i)

        # simulation.run(N_STEPS=1,visualize=False)
        
        p_o, rates, srouting, srouting_tuple = solver.get_prim_objective(with_rates=True)
        print(f"p_o: {p_o:.3f}, rates: {rates.mean():.3f}")
        LOG_OBJ._add_np_log("p_o", i, np.array([p_o]))
        
LOG_OBJ.save_np(LOG_DIR,"final")