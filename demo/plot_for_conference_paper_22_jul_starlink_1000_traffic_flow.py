#!/usr/bin/env python3
"""
Simulation of Starlink satellites with Earth rotation and Vispy visualization.

This script loads Starlink TLE data, filters invalid satellites, computes Earth’s rotation,
and visualizes both the Earth (with a textured sphere) and satellites in a 3D scene.
"""

import math
import os
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

from sim_src.util import GET_FILE_NAME_FOR_SIM_SCRIPT, GET_LOG_PATH_FOR_SIM_SCRIPT, STATS_OBJECT
from working_dir_path import get_working_dir_path

np.set_printoptions(precision=4, suppress=True)

class Demo(Simulation):
    def run_step(self):
        time.sleep(0.01)  # Simulate some processing time
        data_source, data_target = self.solver.data_source, self.solver.data_target
        print(data_source)

        satp = np.column_stack((data_source, data_target)).astype(np.int32)

        self._update_traffic_flow(satp=satp, edge_weight=None, viz=self.viz_list[0])
        capacity = self.solver.compute_capacity(
            np.linalg.norm(self.positions[self.solver.possible_sat_pair_expanded[:, 0]] - self.positions[self.solver.possible_sat_pair_expanded[:, 1]], axis=1)
        )        
        edge_weights_pr = capacity
        weighted_edges = np.column_stack((self.solver.possible_lct_pair_expanded, edge_weights_pr))
        matching = greedy_max_weight_matching(weighted_edges)
        m = len(matching)
        flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
                                  for x in pair), dtype=int, count=2*m)
        connected_lct = flat_array.reshape(-1, 2)
        connected_sat = connected_lct // self.N_LCT_PER_SAT        
        
        capacity = self.solver.compute_capacity(np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1))
        
        capacity = capacity/ capacity.max()  # Normalize capacity for visualization
        capacity = np.log1p(capacity*20)  # Apply log1p for better visualization
        capacity = capacity/ capacity.max()  # Normalize again after log1p
        self._update_o_lisl(satp=connected_sat, viz=self.viz_list[0], edge_weight=capacity/capacity.max())

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
        self.viz_list[0]['text_title'].text = f"" 
        
        a_from = np.tile(self.positions, (self.N_LCT_PER_SAT, 1))
        a_to = np.concatenate((self.front, self.back, self.right, self.left), axis=0) * 0.02
        a_to_left = rotate_deg_in_vector_element_wise(a_to, np.ones(a_to.shape[0]) * self.FOR_THETA_HALF, a_from) + a_from
        a_to_right = rotate_deg_in_vector_element_wise(a_to, -np.ones(a_to.shape[0]) * self.FOR_THETA_HALF, a_from) + a_from
        a_data = np.concatenate((a_from, a_to_left, a_to_right), axis=1).reshape(-1, 3)

        GREY_COLOR = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        num_arrows = self.positions.shape[0] * 12
        arrow_color = np.zeros((num_arrows, 4))
        arrow_color[: num_arrows // self.N_LCT_PER_SAT, :] = GREY_COLOR
        arrow_color[num_arrows // self.N_LCT_PER_SAT: num_arrows // 2, :] = GREY_COLOR
        arrow_color[num_arrows // 2: 3 * num_arrows // self.N_LCT_PER_SAT, :] = GREY_COLOR
        arrow_color[3 * num_arrows // self.N_LCT_PER_SAT:, :] = GREY_COLOR
        arrow_color[:, 3] = np.tile(np.array([1, 0, 0], dtype=np.float32), (self.positions.shape[0] * self.N_LCT_PER_SAT, 1)).reshape(-1)

        if self.lct_mask is not None:
            # Apply the LCT mask to the arrows.
            a_data = a_data[np.repeat(self.lct_mask.transpose().reshape(-1), 3).astype(bool), :]
            arrow_color = arrow_color[np.repeat(self.lct_mask.transpose().reshape(-1), 3).astype(bool), :]

        faces = np.arange(a_data.shape[0]).reshape(-1, 3)

        self.viz_list[0]['triangle'].set_data(vertices=a_data, faces=faces, vertex_colors=arrow_color)
        
        
    def setup_visualization(self):
        self.canvas, self.viz_list = setup_viz_list_one_canvas(sceen_size=(1200, 800), shape=(1, 1))
        for viz in self.viz_list:
            viz['axes'].visible = False
            viz['view'].camera.azimuth = self.compute_rotation() + 90
            viz['view'].camera.elevation = 45
            viz['view'].camera.distance = 2.5
        traffic_flow = scene.visuals.Arrow()
        self.viz_list[0]['view'].add(traffic_flow)
        self.viz_list[0]['traffic_flow'] = traffic_flow     
        
        self.viz_list[0]['arrow'].visible = True
        self.viz_list[0]['p_lisl'].visible = False
        self.viz_list[0]['o_lisl'].visible = False
        self.viz_list[0]['sphere_visual'].visible = False

        triangle = scene.visuals.Mesh(shading=None)
        self.viz_list[0]['view'].add(triangle)
        self.viz_list[0]['triangle']= triangle

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
tle_file_path = os.path.join(get_working_dir_path(),'starlink_16_jul_2025_1600.tle')
ts, valid_satellites, sat_array = load_url_tle_data(tle_file_path, reload=True)

LOG_OBJ = STATS_OBJECT()
LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
i = 1
rng = np.random.default_rng(seed=i)
idx = rng.choice(np.arange(len(valid_satellites)), size=1000, replace=False)
valid_satellites = [valid_satellites[x] for x in idx]
models = [sat.model for sat in valid_satellites]
sat_array = SatrecArray(models)
Simulation.FRONT_COLOR = np.array([0, 0.5, 1, 1])
Simulation.BACK_COLOR = np.array([0, 0.5, 1, 1])
Simulation.RIGHT_COLOR = np.array([0, 0.5, 1, 1])
Simulation.LEFT_COLOR = np.array([0, 0.5, 1, 1])

simulation = Demo(ts, sat_array)
print(simulation.FRONT_COLOR, simulation.BACK_COLOR, simulation.RIGHT_COLOR, simulation.LEFT_COLOR)
simulation.PLOT_POTENTIAL_LISL = True
simulation.PLOT_POTENTIAL_LISL_LINE_WIDTH = 1
simulation.config_l_mask()
simulation.update_space()

solver = mr_solver()
simulation.set_solver(solver)
simulation.update_solver_traffic_info(seed=i)

simulation.run(TOT_STEPS=50000,visualize=True)