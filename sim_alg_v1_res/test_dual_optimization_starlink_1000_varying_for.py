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
import plotext

from sim_src.util import CSV_WRITER_OBJECT, GET_LOG_PATH_FOR_SIM_SCRIPT, STATS_OBJECT, counted, p_true

np.set_printoptions(precision=4,threshold=10,linewidth=80, edgeitems=2)

class lpdsolver(mr_solver):
    def update_step_rates_prices(self):
        self.N_STEP += 1
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

class DualSimulation(Simulation):
    def get_simulation_time(self):
        # return time at 2025 jun 1st
        return self.ts.utc(2025, 7, 16, 16, 0, 0)

    def run_step(self):
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
        s_t = np.column_stack((self.solver.data_source, self.solver.data_target))
        
        gap = p_o - d_o
        # Compute the gap.
        print(f"Gap: {gap:.3f}, Exp: {np.exp(-(gap)):.3f}, d_o: {d_o:.3f}, p_o: {p_o:.3f}")
        avg_p_o = self._moving_average("p_o", p_o)
        print(f"avg: {avg_p_o}")
        
        self._add_np_log("gap", self.N_STEP, np.array([gap]))
        self._add_np_log("d_o", self.N_STEP, np.array([d_o]))
        self._add_np_log("p_o", self.N_STEP, np.array([p_o]))
        
        
        if self.VISUALIZE:
            dual_rates = self.solver.s_t_traffic_rates
            self._update_traffic_flow(satp=s_t, edge_weight=dual_rates/20, viz=self.viz_list[3])
            
            connected_sat, connected_lct = self.solver.get_dual_matching()
            capacity = self.solver.compute_capacity(
                np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
            )
            self._update_o_lisl(satp=connected_sat, viz=self.viz_list[1], edge_weight=capacity/capacity.max())

            self._update_o_lisl(satp=srouting[:,0:2].astype(np.int64), edge_weight=None, viz=self.viz_list[0]) 

            vis_prices = edge_prices[:,2]/np.max(edge_prices[:,2]+1e-10)
            self._update_o_lisl(satp=edge_prices[:,0:2].astype(np.int64), edge_weight=vis_prices, viz=self.viz_list[2]) 
        
            self.viz_list[0]['text_title'].text = f"Weighted Shortest Path Routing\n - Average Hops {lengths.mean():.2f}" 
            self.viz_list[1]['text_title'].text = f"Maximum Weight Laser Matching\n - Average Per Link Rate (Gbps) {capacity.mean():.2f}" 
            self.viz_list[2]['text_title'].text = f"Laser Link Pricing (Dual Variables)\n - Average Price (Gbps Per Hop) {edge_prices[:,2].mean():.2f}"
            self.viz_list[3]['text_title'].text = f"Source-to-Target Traffic Flow\n - Average Rate (Gbps) {dual_rates.mean():.2f}"
        
    
        p_o_mwm, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = self.solver.get_prim_objective_mwm(with_rates=True)
        self._printalltime(f"Prim objective: {p_o}, MWM: {p_o_mwm}, Ratio: {p_o/p_o_mwm:.3f}")
        self._add_np_log("mwm", self.N_STEP, np.array([p_o_mwm]))
        self._add_np_log("ratio", self.N_STEP, np.array([p_o/p_o_mwm]))
    
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

        # Load tle data.
from working_dir_path import get_working_dir_path
import os

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)
tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')

for n_sat in [1000]:
    for ratio in [0.]:
        for FOR in [30, 40, 50, 60, 70, 80, 90]:
            for i in range(1):
                # Create simulation instance.
                ts, valid_satellites, sat_array = generate_tle_partly_regular_constellation1000(n_sat=n_sat, ratio=ratio, starlink_tle_path=tle_file_path, seed=i)
                DualSimulation.FOR_THETA_HALF = FOR
                simulation = DualSimulation(ts, sat_array)
                simulation.config_l_mask(seed=i)
                simulation.update_space()

                solver = lpdsolver()
                simulation.set_solver(solver)
                simulation.run(TOT_STEPS=500,visualize=False)
                
                p_o_list = []
                p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = simulation.solver.get_prim_objective(with_rates=True)
                p_o_list.append(p_o)
                for m, matching_method in enumerate(["grid", "maxc", "rand"]):
                    for r, routing_method in enumerate(["ospf"]):
                        p_o_heu, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = simulation.solver.get_prim_objective_heuristic(with_rates=True,
                            matching_method=matching_method, routing_method=routing_method, seed=i
                        )
                        p_o_list.append(p_o_heu)
                res_list = [n_sat, FOR] + p_o_list
                LOG_CSV_WRITTER.log_mul_scalar("res.csv", i, res_list)
