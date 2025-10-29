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

from sim_alg_j1_res.train_rl_starlink_1000_ld import GNNSimulation


class ldl_sg_compare_solver(mr_solver):
    GNN_INITIALIZED = False
    def load_gnn(self, path):
        print("Initializing GNN model")
        self.model = ld_model()
        self.model.load_model(path)
        self.model.eval()
        
    def infer_gnn(self):
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

        new_prices = self.model.get_output_np_edge_weight(x, possible_sat_pair_non_expanded_sym, possible_capacity_non_expanded_sym, use_target=False)

        self.price_graph.price_graph = sp.csr_matrix((new_prices, (possible_sat_pair_non_expanded_sym[:, 0], possible_sat_pair_non_expanded_sym[:, 1])), shape=(self.n_sat, self.n_sat))
        
    @counted
    def update_step_rates_prices(self):
        step_size = self.ALPHA / (self.N_STEP ** self.BETA)
        self._print("Updating edge prices")
        connected_sat, connected_lct = self.get_dual_matching()
        self._print(f"Connected sat: {connected_sat.shape}, Connected lct: {connected_lct.shape}")
        # qx_csr = sp.csr_matrix((self.n_sat, self.n_sat))

        costs, lengths, paths_all = self.get_dual_srouting()
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )
        
        self._print(f"Maximize rate: {srouting.shape}")            
        # self.s_t_traffic_rates -= step_size * (-1 + costs)
        # self.s_t_traffic_rates = np.clip(self.s_t_traffic_rates, 0, None)
        self._printalltime(f"starting to compute rates")
        tic = self._get_tic()
        self.s_t_traffic_rates = self.get_rates_dual(costs=costs)
        tim = self._get_tim(tic)
        self._printalltime(f"Computed rates: {self.s_t_traffic_rates.shape}, Time: {tim:.4f} us")

        self._print(f"Appr rate: MAX: {np.max(self.s_t_traffic_rates)}, MIN: {np.min(self.s_t_traffic_rates)}")
        self._print(f"Cost path: MAX: {np.max(costs)}, MIN: {np.min(costs)}")
        
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, self.s_t_traffic_rates, lengths, paths_all
        )
        self._print(f"Compute qx: {srouting.shape}")
        if np.asarray(srouting).size == 0:
            qx_csr = sp.csr_matrix((self.n_sat, self.n_sat))
        else:
            traffic = srouting[:,4]
            self._print(f"Tr: max: {np.max(traffic)}, min: {np.min(traffic)}")
            qx = build_csr(self.n_sat, srouting[:,0:2], traffic, merging_method='sum')
            self._print(f"qx: max: {np.max(qx[2])}, min: {np.min(qx[2])}")
            qx_csr = sp.csr_matrix((qx[2], qx[1], qx[0]), shape=(self.n_sat, self.n_sat))

        self._print(f"Compute rc: {connected_sat.shape}")
        if np.asarray(connected_sat).size == 0:
            rc_csr = sp.csr_matrix((self.n_sat, self.n_sat))
        else:
            edge_in_both_direction = np.concatenate((connected_sat, connected_sat[:, ::-1]), axis=0)
            capacity_matched = self.compute_capacity(
                np.linalg.norm(self.positions[edge_in_both_direction[:, 0]] - self.positions[edge_in_both_direction[:, 1]], axis=1)
            )
            self._print(f"Cm: max: {np.max(capacity_matched)}, min: {np.min(capacity_matched)}")
            rc = build_csr(self.n_sat, edge_in_both_direction, capacity_matched, merging_method='sum', sym_half=False)
            self._print(f"rc: max: {np.max(rc[2])}, min: {np.min(rc[2])}")
            rc_csr = sp.csr_matrix((rc[2], rc[1], rc[0]), shape=(self.n_sat, self.n_sat))
        
        # Update edge prices
        edge_in_both_direction = np.concatenate((self.possible_sat_pair_expanded, self.possible_sat_pair_expanded[:, ::-1]), axis=0)
        
        old_price = self.price_graph.get_prices(edge_in_both_direction)
        self._print(f"Oe: max: {np.max(old_price)}, min: {np.min(old_price)}")
        
        self._print(f"diff: {(qx_csr>rc_csr).mean()}")
        
        dif_price_graph = step_size * (qx_csr - rc_csr)
        
        self.price_graph.add_prices(dif_price_graph)     
        new_price = self.price_graph.get_prices(edge_in_both_direction)
        self._print(f"Ne: max: {np.max(new_price)}, min: {np.min(new_price)}")
        
        self._print(f"Sz: {step_size}")
        plotext.title("Prices Distribution")
        plotext.hist(np.log10(old_price+1e-5), bins=50, norm=True)
        plotext.plotsize(100, 30)
        plotext.xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3],)
        plotext.show()
        plotext.clf()        
        return new_price    

if __name__ == "__main__":
    N_CONSTELLATION = 100
    SG_STEPS = 500
            
    # Load tle data.
    from working_dir_path import get_working_dir_path
    import os
    tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')

    LOG_OBJ = STATS_OBJECT()
    LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

    LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)

    for seed in range(N_CONSTELLATION):
        solver = ldl_sg_compare_solver()
        ts, valid_satellites, sat_array = generate_tle_partly_regular_constellation1000(n_sat=1000, ratio=0.0, starlink_tle_path=tle_file_path, seed=seed)
        simulation = GNNSimulation(ts, sat_array)
        simulation.config_l_mask(seed=seed)
        simulation.update_space()
        simulation.set_solver(solver)
        tic = LOG_OBJ._get_tic()
        for step in range(SG_STEPS):
            print(f"Running simulation with step: {step}")
            ratio, p_o, d_o, p_o_mwm = simulation.run_step()
            tim = LOG_OBJ._get_tim(tic,remove_timer=False)
            LOG_CSV_WRITTER.log_mul_scalar("sg", step, [tim,ratio, p_o, d_o, p_o_mwm], g_step=seed)

        PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "selected_nn/ld_model.model_final_beta_0_7000_pt.pt")
        solver.load_gnn(path=PATH)
        tic = LOG_OBJ._get_tic()
        solver.infer_gnn()
        tim = LOG_OBJ._get_tim(tic,remove_timer=True)
        d_o, edge_prices = solver.get_dual_objective(with_prices=True)
        p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = solver.get_prim_objective(with_rates=True)
        LOG_CSV_WRITTER.log_mul_scalar("ldl", step, [tim, ratio, p_o, d_o, p_o_mwm], g_step=seed)
