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

class DualSimulation(Simulation):
    def set_solver(self, solver):
        self.solver:mr_solver = solver
        self.solver.init_constellation(self.filtered_repeated, self.filtered_expanded, self.positions)

    def run_step(self):
        # Check the constellation connectivity
        connected, comp = self.solver.check_connected()
        print(f"Constellation is connected: {connected}")
         
        
        # Evaluate g(lambda)
        d_o = self.solver.get_dual_objective()
        # Compute the prim p.
        p_o = self.solver.get_prim_objective()
        
        # Compute the gap.
        print(f"Gap: {p_o - d_o}, Exp: {np.exp(-(p_o - d_o))}, d_o: {d_o}, p_o: {p_o}")
        
        self.solver.update_step_rates_prices()
        # self.srouting = self.solver.get_dual_srouting()
        # logger.debug(f"Routing shape: {self.srouting.shape}")
        # self._update_o_lisl(satp=self.srouting[:,0:2].astype(np.int64), edge_weight=None, viz=self.viz_list[1]) 
        
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
ts, valid_satellites, sat_array = generate_walker_constellation_add_planes()

# Create simulation instance.
simulation = DualSimulation(ts, sat_array)


solver = mr_solver()

solver._debug()
simulation.set_solver(solver)

simulation.run()

