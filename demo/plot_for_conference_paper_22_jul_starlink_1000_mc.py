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
import vispy.io as io
from vispy.gloo.util import _screenshot

import logging
import plotext

from sim_src.util import CSV_WRITER_OBJECT, GET_FILE_NAME_FOR_SIM_SCRIPT, GET_LOG_PATH_FOR_SIM_SCRIPT, STATS_OBJECT, counted, p_true

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
            connected_sat, connected_lct = self.solver.get_dual_matching()
            capacity = self.solver.compute_capacity(
                np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
            )
            # self._update_o_lisl(satp=connected_sat, viz=self.viz_list[0], binary=True, color=np.array([0.5, 0.5, 1.], dtype=np.float32), width=3)
            # self._update_o_lisl(satp=srouting[:,0:2].astype(np.int64), viz=self.viz_list[1], binary=True, width=3)

            p_o_heu, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = simulation.solver.get_prim_objective_heuristic(with_rates=True,
                            matching_method="mc", routing_method="ospf", seed=i
            )
            self._update_o_lisl(satp=connected_sat, viz=self.viz_list[0], binary=True, color=np.array([0.5, 0.5, 1.], dtype=np.float32), width=3)
            self._update_o_lisl(satp=srouting[:,0:2].astype(np.int64), viz=self.viz_list[1], binary=True, width=3)
            
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
        
        a_from = np.tile(self.positions, (self.N_LCT_PER_SAT, 1))
        a_to = np.concatenate((self.front, self.back, self.right, self.left), axis=0) * 0.1
        a_to_left = rotate_deg_in_vector_element_wise(a_to, np.ones(a_to.shape[0]) * self.FOR_THETA_HALF, a_from) + a_from
        a_to_right = rotate_deg_in_vector_element_wise(a_to, -np.ones(a_to.shape[0]) * self.FOR_THETA_HALF, a_from) + a_from
        a_data = np.concatenate((a_from, a_to_left, a_to_right), axis=1).reshape(-1, 3)

        GREY_COLOR = np.array([0.3, 0.5, 0.65, 0.5], dtype=np.float32)
        num_arrows = self.positions.shape[0] * 12
        arrow_color = np.zeros((num_arrows, 4))
        arrow_color[: num_arrows // self.N_LCT_PER_SAT, :] = GREY_COLOR
        arrow_color[num_arrows // self.N_LCT_PER_SAT: num_arrows // 2, :] = GREY_COLOR
        arrow_color[num_arrows // 2: 3 * num_arrows // self.N_LCT_PER_SAT, :] = GREY_COLOR
        arrow_color[3 * num_arrows // self.N_LCT_PER_SAT:, :] = GREY_COLOR
        arrow_color[:, 3] = np.tile(np.array([1, 0.25, 0.25], dtype=np.float32), (self.positions.shape[0] * self.N_LCT_PER_SAT, 1)).reshape(-1)

        if self.lct_mask is not None:
            # Apply the LCT mask to the arrows.
            a_data = a_data[np.repeat(self.lct_mask.transpose().reshape(-1), 3).astype(bool), :]
            arrow_color = arrow_color[np.repeat(self.lct_mask.transpose().reshape(-1), 3).astype(bool), :]

        faces = np.arange(a_data.shape[0]).reshape(-1, 3)

        self.viz_list[0]['triangle'].set_data(vertices=a_data, faces=faces, vertex_colors=arrow_color)
        
        
    def setup_visualization(self):
        screen_size=(1200, 800)
        shape=(2, 1)
        ret = []
        canvas = scene.SceneCanvas(title='Mega-Constellation Simulation',size=screen_size, position=(0, 0),
                keys='interactive', show=True, bgcolor=(1.0, 1.0, 1.0, 1.0))
        grid = canvas.central_widget.add_grid()
        camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, elevation=45, distance=3)

        for i in range(shape[0]):
            for j in range(shape[1]):
                view = grid.add_view(row=i, col=j)
                view.pos = (j * (screen_size[0] // shape[1]), i * (screen_size[1] // shape[0]))
                view.size = (screen_size[0] // shape[1], screen_size[1] // shape[0])
                view.camera = camera
                if i == 0 and j == 0:
                    # Setup the first view with Earth texture
                    ret.append(setup_visualization(view=view, idx=(i, j)))
                else:
                    ret.append(setup_visualization(view=view, idx=(i, j), earth_texture="simple_earth_texture.png"))
        self.canvas, self.viz_list = canvas, ret
        
        
        for viz in self.viz_list:
            viz['axes'].visible = False
            viz['view'].camera.azimuth = self.compute_rotation() + 90
            viz['view'].camera.elevation = 50
            viz['view'].camera.distance = 2.5
            viz['text_title'].text = f"" 


        self.viz_list[0]['arrow'].visible = False
        self.viz_list[0]['o_lisl'].visible = True
        self.viz_list[1]['o_lisl'].visible = True

        triangle = scene.visuals.Mesh(shading=None)
        self.viz_list[0]['view'].add(triangle)
        self.viz_list[0]['triangle']= triangle
        
        self.viz_list[0]['scatter'].scaling = False

        self.viz_list[0]['scatter'].scaling = False
        
        self.viz_list[1]['o_scatter'].visible = False

        self.canvas.events.mouse_double_click.connect(self.handle_double_click)

    def handle_double_click(self, event):
        """
        Handle double-click events to toggle visibility of the sphere visual.
        """
        print("Double-click detected")
        self.canvas.update()
        self.canvas.show()
        img = _screenshot(viewport=(0,0, self.canvas.physical_size[0], self.canvas.physical_size[1]), alpha=True)
        path = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
        try:
            os.mkdir(path)
        except:
            pass
        path = os.path.join(path, GET_FILE_NAME_FOR_SIM_SCRIPT(__file__) + f"_{self.N_STEP:05d}.png")
        io.write_png(path,img)

        # Load tle data.
from working_dir_path import get_working_dir_path
import os

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
LOG_CSV_WRITTER = CSV_WRITER_OBJECT(path=LOG_DIR)
tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')

for n_sat in [1000]:
    for ratio in [0.]:
        for i in [4]:
            # Create simulation instance.
            ts, valid_satellites, sat_array, shell_metadata = generate_tle_starlink_shell_block_constellation(
                n_sat=n_sat,
                starlink_tle_path=tle_file_path,
            )
            simulation = DualSimulation(ts, sat_array)
            simulation.config_l_mask(seed=i)
            simulation.update_space()

            solver = lpdsolver()
            simulation.set_solver(solver)
            simulation.run(TOT_STEPS=2000,visualize=True)
            
