#!/usr/bin/env python3

import math
import time
from turtle import width

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
import vispy.io as io
from vispy.gloo.util import _screenshot

import logging

from sim_src.util import CSV_WRITER_OBJECT, GET_LOG_PATH_FOR_SIM_SCRIPT, STATS_OBJECT, counted, GET_FILE_NAME_FOR_SIM_SCRIPT
from working_dir_path import get_working_dir_path

import torch
torch.set_float32_matmul_precision('medium')

np.set_printoptions(precision=4, suppress=True)

from sim_alg_j1_res.test_ld_starlink_1000_sg_compare import ldl_sg_compare_solver

from sim_alg_j1_res.test_ld_starlink_1000_constellation_time_varying import Time_Varying_Simulation

import os
class Visual_Time_Varying_Simulation(Time_Varying_Simulation):
    def setup_visualization(self):
        self.canvas, self.viz_list = setup_viz_list_one_canvas(screen_size=(1200, 800), shape=(1, 1))
        self.viz_list[0]['text_title'].text = f"" 
        self.viz_list[0]['arrow'].visible = False
        self.viz_list[0]['o_lisl'].visible = True
        self.viz_list[0]['sphere_visual'].visible = True
        self.viz_list[0]['o_scatter'].visible = True
        
        self.viz_list[0]['t_lisl'] = scene.visuals.Arrow()
        self.viz_list[0]['view'].add(self.viz_list[0]['t_lisl'])

        triangle = scene.visuals.Mesh(shading=None)
        self.viz_list[0]['view'].add(triangle)
        self.viz_list[0]['triangle']= triangle
        self.viz_list[0]['triangle'].visible = True

        
        
    def camera_rotate_to_0_lat(self):
        for viz in self.viz_list:
            viz['axes'].visible = False
            viz['view'].camera.azimuth = self.compute_rotation() + 90
            viz['view'].camera.elevation = 30
            viz['view'].camera.distance = 2.5

    def save_img(self,path, name=""):
        self.canvas.update()
        self.canvas.show()
        from vispy import app as _app
        _app.process_events()
        img = _screenshot(viewport=(0,0, self.canvas.physical_size[0], self.canvas.physical_size[1]), alpha=True)
        try:
            os.mkdir(path)
        except:
            pass
        path = os.path.join(path, f"_{self.N_STEP:05d}_{name}.png")
        io.write_png(path,img)

    def update_connectable_lct_pairs(self, connectable_lct_pairs, active_indicators):
        colors = np.ones((connectable_lct_pairs.shape[0], 4), dtype=np.float32)
        colors[~active_indicators, :] = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)
        colors[active_indicators, :] = np.array([0.1, 0.7, 1., 0.5], dtype=np.float32)
        connectable_lct_pairs_sat = connectable_lct_pairs // self.N_LCT_PER_SAT
        print(connectable_lct_pairs_sat)
        print(f"Updating {connectable_lct_pairs_sat.shape[0]} links in visualization.", colors.shape)
        self._update_o_lisl(satp=connectable_lct_pairs_sat, viz=self.viz_list[0], color=colors, width=1.0)

    def update_connected_lct_pairs(self, connected_lct_pairs, active_indicators):
        colors = np.ones((connected_lct_pairs.shape[0], 4), dtype=np.float32)
        colors[~active_indicators, :] = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)
        colors[active_indicators, :] = np.array([0.1, 0.7, 1., 1], dtype=np.float32)
        connected_lct_pairs = connected_lct_pairs // self.N_LCT_PER_SAT
        print(connected_lct_pairs)
        print(f"Updating {connected_lct_pairs.shape[0]} links in visualization.", colors.shape)
        self._update_t_lisl(satp=connected_lct_pairs, viz=self.viz_list[0], color=colors, width=5.0)
    
    def _update_t_lisl(self, satp = None, edge_weight=None, viz=None, binary=False, color=None, width=5):
        # Update optional LISL lines.
        if np.asarray(satp).size == 0:
            return
        if viz is not None:
            edge_from = self.positions[satp[:, 0]]
            edge_to = self.positions[satp[:, 1]]
            if edge_weight is None:
                edge_weight = np.ones(satp.shape[0], dtype=self.EDGE_COLOR.dtype)*0.5

            edge_weight = edge_weight.reshape(-1, 1)
            edge_from = edge_from * (1 + 0.0001 * edge_weight)
            edge_to = edge_to * (1 + 0.0001 * edge_weight)
            
            o_lisl_data = np.concatenate((edge_from, edge_to), axis=1).reshape(-1, 3)
            if not binary:
                edges_color_data = np.zeros((o_lisl_data.shape[0], 4), dtype=self.EDGE_COLOR.dtype)
                if edge_weight is not None:
                    edges_color_data[:, 3] = np.concatenate((edge_weight, edge_weight), axis=1).reshape(-1)
                else:
                    edges_color_data[:, 3] = 1
            else:
                edges_color_data = np.zeros((o_lisl_data.shape[0], 4), dtype=self.EDGE_COLOR.dtype)
                edges_color_data[:, 3] = 1
                edge_weight_red = (edge_weight > 0.5).astype(np.float32)
                edge_weight_green = (edge_weight < 0.5).astype(np.float32)
                edges_color_data[:, 0] = np.concatenate((edge_weight_red, edge_weight_red), axis=1).reshape(-1)
                edges_color_data[:, 1] = np.concatenate((edge_weight_green, edge_weight_green), axis=1).reshape(-1)
            
            if color is not None:
                if color.shape[1] == 3:
                    edges_color_data[:, :3] = np.tile(color, (1, 2)).reshape(-1, 3)
                elif color.shape[1] == 4:
                    edges_color_data[:, :4] = np.tile(color, (1, 2)).reshape(-1, 4)
                else:
                    raise ValueError("Color array must have shape (N, 3) or (N, 4).")    
            viz['t_lisl'].set_data(pos=o_lisl_data, color=edges_color_data, width=width, connect='segments')
        
    def update_for(self):
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
        

if __name__ == "__main__":
    SG_STEPS = 500
    N_CONSTELLATION = 1
    
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
        init_simulation = Visual_Time_Varying_Simulation(ts, sat_array)
        init_simulation.setup_visualization()
        init_simulation.reset_simulation_time()
        init_simulation.config_l_mask(seed=seed)
        init_simulation.update_space()
        init_simulation.set_solver(solver)
        init_simulation.update_solver_traffic_info(seed=seed)
        PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "selected_nn/ld_model.model_final_beta_0_7000_pt.pt")
        solver.load_gnn(path=PATH)
        solver.infer_gnn()
        p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = solver.get_prim_objective(with_rates=True)
        active_indicators = np.ones(connected_lct.shape[0], dtype=bool)
        init_simulation.update_connected_lct_pairs(connected_lct, active_indicators)
        connectable_lct_pairs = solver.possible_lct_pair_expanded
        active_indicators = np.ones(connectable_lct_pairs.shape[0], dtype=bool )
        init_simulation.update_connectable_lct_pairs(connectable_lct_pairs, active_indicators)
        init_simulation.update_for()
        init_simulation.camera_rotate_to_0_lat()
        init_simulation.visualize()
        init_simulation.save_img(LOG_DIR)
        init_simulation.N_STEP += 1
        init_simulation.step_time_us(22e3)
        init_simulation.update_space()
        init_simulation.set_solver(solver)
        init_simulation.update_solver_traffic_info(seed=seed)

        # init_simulation.setup_visualization()
        connectable_lct_pairs_changed = solver.possible_lct_pair_expanded
        mask = (connectable_lct_pairs[:, None, :] == connectable_lct_pairs_changed[None, :, :]).all(-1).any(1)
        init_simulation.update_connectable_lct_pairs(connectable_lct_pairs, mask)

        mask = (connected_lct[:, None, :] ==  connectable_lct_pairs_changed[None, :, :]).all(-1).any(1)
        init_simulation.update_connected_lct_pairs(connected_lct, mask)
        init_simulation.update_for()
        init_simulation.camera_rotate_to_0_lat()
        init_simulation.visualize()
        init_simulation.save_img(LOG_DIR)
        
        init_simulation.close_app()