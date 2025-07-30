import math
import time

from threading import Thread
import queue
import psutil
import numpy as np
from scipy.spatial import cKDTree

from skyfield.api import load
from skyfield.sgp4lib import TEME

from sim_mld.terrain import terrain
from sim_mld.visual import *
from sim_mld.tle import *
from sim_mld.constellation import *

from sim_mld.solver import mr_solver

from sim_src.util import STATS_OBJECT, counted


def khot_matrix(n_rows, n_cols, k, rng=None, seed=None, dtype=np.float32):
    """
    Return an (n_rows × n_cols) array in which each row contains exactly `k`
    ones (aka a k-hot encoding).

    Parameters
    ----------
    n_rows : int
    n_cols : int
    k      : int          # 1 ≤ k ≤ n_cols
    replace : bool        # sample with/without replacement
    seed    : int | None  # reproducible RNG seed
    dtype   : np.dtype    # 0/1 storage type (bool, int8, etc.)
    """
    if not 0 < k <= n_cols:
        raise ValueError("k must be between 1 and n_cols")
    if rng is None:
        rng = np.random.default_rng(seed)
    order  = rng.permuted(np.tile(np.arange(n_cols), (n_rows, 1)), axis=1)
    chosen = order[:, :k]

    mat = np.zeros((n_rows, n_cols), dtype=dtype)
    mat[np.arange(n_rows)[:, None], chosen] = 1
    return mat



class Simulation(STATS_OBJECT):
    FOR_THETA_HALF: float = 60.0  # Angle in degrees for the satellite LT direction.
    LISL_MAX_DISTANCE: float = 3000.0  # Maximum distance for LISL in km.
    TIME_SCALE: float = 15.0
    EARTH_RADIUS: float = 6371.0  # Earth's radius in km.
    
    N_LCT_PER_SAT: int = 4  # Number of LCTs per satellite.
        
    PLOT_POTENTIAL_LISL: bool = False 
    PLOT_POTENTIAL_LISL_LINE_WIDTH: float = 0.02
    PLOT_SATELLITE_LCTS: bool = True
    
    PLOT_GWS_POINT_SIZE: float = 10
    PLOT_SAT_POINT_SIZE: float = 10
    
    PLOT_TEXT: bool = False
    
    FRONT_COLOR = np.array([0, 0, 0.85, 1])
    BACK_COLOR = np.array([0.1, 0.6, 0.1, 1])
    RIGHT_COLOR = np.array([1, 0, 0, 1])
    LEFT_COLOR = np.array([0.75, 0.75, 0, 1])
    

    VISUALIZE: bool = False
    def __init__(self, ts, sat_array):
        """
        Initialize the simulation.

        Parameters:
            ts: Skyfield timescale.
            sat_array: Vectorized satellite propagation array.
        """
        self.ts = ts
        self.earth_rotation_angle = 0.0
        self.terrain:terrain = terrain()  # Placeholder for terrain data.
        
        self.sat_array = sat_array

        self.n_sat = len(sat_array)
        self.sat_cKDtree = None
        
        self.canvas = None
        self.viz_list = []
               
        self.simulation_start_time = self.ts.now()
        self.real_start_time = time.perf_counter()
        self.current_time = self.update_simulation_time()
        self.TOT_STEPS = 0
        self.update_count = 0
        self.accumulated_update_time = 0
        self.average_update_time = 1

        # Initialize satellite positions and velocities.
        self.positions = None
        self.velocities = None
        
        # Initialize LCT on indicator.
        self.lct_mask = None
        
        # Initialize directional vectors.
        self.front = None
        self.back = None
        self.down = None
        self.right = None
        self.left = None
        
        # Initialize LCT directions
        self.lct_directions = None
        
        # Initialize p_satp edges and directions.
        self.edges = None
        
        # Initialize view stacks.
        self.view_from_stack = None
        self.view_to_stack = None
        
        # Initialize potential LISL edges and view stacks.
        self.filtered_edges = None
        self.filtered_sat_pair_repeated = None
        self.filtered_lct_pair_expanded = None
        
        # Precompute cosine threshold once.
        self.cos_threshold = math.cos(math.radians(self.FOR_THETA_HALF))
        self.view_LT_pair_min_cos = None
        
        # Initialize the connected satellite and LISL data.
        self.connected_sat = None
        self.connected_lct = None

        # Initialize the routing
        self.srouting = None

        self.profiled_time = {}       
                
        self.solver = None
        
        self.data_thread:Thread = None
        self.data_thread_start_time = None
        self.data_thread_last_tickT = None
        self.data_thread_return_queue:queue.Queue = queue.Queue()

    def set_solver(self, solver):
        """
        Set the solver for the simulation.

        Parameters:
            solver: The solver instance to be used.
        """
        self.solver:mr_solver = solver
        self.update_solver_constellation_info()
    
    def set_terrain(self, terrain:terrain):
        assert isinstance(terrain, terrain), "terrain must be an instance of the terrain class"
        self.terrain = terrain
    
    def update_solver_constellation_info(self):
        # Initialize the solver with the current constellation data.
        self.solver.n_sat = self.n_sat
        self.solver.positions = self.positions        
        self.solver.possible_sat_pair_expanded = self.filtered_sat_pair_repeated
        self.solver.possible_lct_pair_expanded = self.filtered_lct_pair_expanded
        self.solver.possible_lct_pair_expanded_view_cos = self._get_view_cos()
        
        self.solver._reset_price_graph(0.)
        
    def update_solver_traffic_info(self, seed=0):
        self.solver.data_source, self.solver.data_target, self.solver.forward_traffic_capacity, self.solver.forward_traffic_demand = self.terrain.get_traffic_info_test(self.positions,seed=seed)
        self.solver.s_t_traffic_rates = np.zeros(self.solver.data_source.shape[0], dtype=np.float32)
        self._printalltime(f"Before s_t pair filtering seed {seed}, source: {self.solver.data_source.shape[0]}, target: {self.solver.data_target.shape[0]}")
        self.solver._remove_non_connected_s_t_pairs()
        self._printalltime(f"Updated traffic info with seed {seed}, source: {self.solver.data_source.shape[0]}, target: {self.solver.data_target.shape[0]}")
        
    def _assign_lct(self):
        self.lct_directions = np.concatenate((self.front, self.back, self.right, self.left), axis=1).reshape(-1, self.N_LCT_PER_SAT, 3)
        
    def setup_visualization(self):
        self.canvas, self.viz_list = setup_viz_list_one_canvas(sceen_size=(1200, 800), shape=(2, 2))

    def update_simulation_time(self):
        elapsed_real = time.perf_counter() - self.real_start_time
        elapsed_scaled = elapsed_real * self.TIME_SCALE
        delta_days = elapsed_scaled / 86400  # Convert seconds to days.
        new_tt_jd = self.simulation_start_time.tt + delta_days
        self.current_time = self.ts.tt(jd=new_tt_jd)
        return self.current_time
    
    def get_simulation_time(self):
        """
        Compute the current simulation time based on the time scaling factor.

        Returns:
            Skyfield Time: The current simulation time.
        """
        return self.current_time

    def compute_rotation(self) -> float:
        """
        Compute the initial rotation angle from the current GMST.

        Returns:
            float: Rotation angle in degrees.
        """
        t_now = self.get_simulation_time()
        gmst_hours = t_now.gmst
        rotation_angle_deg = gmst_hours * 15  # 15° per hour.
        self._print("GMST: %.2f hours, Rotation angle: %.2f degrees", gmst_hours, rotation_angle_deg)
        return rotation_angle_deg

    def _update_earth(self):
        self._print(f'Updating Earth rotation... {self.update_count}')
        """Update the Earth's rotation transformation."""
        self.earth_rotation_angle = self.compute_rotation()
        self.terrain.update_earth_rotation(self.earth_rotation_angle)
        for viz in self.viz_list:
            viz['sphere_visual'].transform.reset()
            viz['sphere_visual'].transform.rotate(self.earth_rotation_angle, (0, 0, 1))
            viz['o_scatter'].set_data(self.terrain.get_ground_station_positions(), face_color=[1, 0.65, 0, 1], size=self.PLOT_GWS_POINT_SIZE, edge_width_rel=0)


    def _update_satellite_positions(self):
        self._print(f'Updating satellite positions and velocities... {self.update_count}')
        """Update satellite positions and velocities."""
        current_time = self.get_simulation_time()
        error_upd, pos_upd, vel_upd = self.sat_array.sgp4(
            np.array([current_time.whole]),
            np.array([current_time.ut1_fraction])
        )
        positions = np.array(pos_upd).reshape(-1, 3) / self.EARTH_RADIUS
        velocities = np.array(vel_upd).reshape(-1, 3) / self.EARTH_RADIUS
        
        R_icrs_to_teme = TEME.rotation_at(current_time)
        R_teme_to_icrs = R_icrs_to_teme.T

        self.positions = rotation_matmul(positions, R_teme_to_icrs)
        self.velocities = rotation_matmul(velocities, R_teme_to_icrs)

        for viz in self.viz_list:
            # Update the satellite markers.
            viz['scatter'].set_data(self.positions, face_color=[0, 0.1, 0, 0.75], size=self.PLOT_SAT_POINT_SIZE, edge_width_rel=0)

    def _update_satellite_arrows(self):
        self._print(f'Updating satellite arrows... {self.update_count}')
        """Update satellite LT direction arrows."""
        self.front, self.back, self.down, self.right, self.left = update_arrows(self.velocities, self.positions)
        if self.PLOT_SATELLITE_LCTS:
            a_from = np.tile(self.positions, (self.N_LCT_PER_SAT, 1))
            a_to = np.concatenate((self.front, self.back, self.right, self.left), axis=0) * 0.005 + a_from
            a_data = np.concatenate((a_from, a_to), axis=1).reshape(-1, 3)

            num_arrows = self.positions.shape[0] * 8
            arrow_color = np.zeros((num_arrows, 4))
            arrow_color[: num_arrows // self.N_LCT_PER_SAT, :] = self.FRONT_COLOR
            arrow_color[num_arrows // self.N_LCT_PER_SAT: num_arrows // 2, :] = self.BACK_COLOR
            arrow_color[num_arrows // 2: 3 * num_arrows // self.N_LCT_PER_SAT, :] = self.RIGHT_COLOR
            arrow_color[3 * num_arrows // self.N_LCT_PER_SAT:, :] = self.LEFT_COLOR
            arrow_color[:, 3] = 1

            if self.lct_mask is not None:
                # Apply the LCT mask to the arrows.
                a_data = a_data[np.repeat(self.lct_mask.transpose().reshape(-1), 2).astype(bool), :]
                arrow_color = arrow_color[np.repeat(self.lct_mask.transpose().reshape(-1), 2).astype(bool), :]

            for viz in self.viz_list:
                # Update the satellite arrows.
                viz['arrow'].set_data(pos=a_data, color=arrow_color, width=self.PLOT_SAT_POINT_SIZE, connect='segments')

    def _update_p_satp(self):
        self._print(f'Updating potential satellite pairs... {self.update_count}')
        """Updating potential satellite pairs using KDTree."""
        
        # Compute the KDTree for efficient nearest neighbor search.
        tic = time.perf_counter()
        self.sat_cKDtree = cKDTree(self.positions)
        distance_threshold = self.LISL_MAX_DISTANCE / self.EARTH_RADIUS
        self.edges = self.sat_cKDtree.query_pairs(r=distance_threshold, output_type='ndarray')
        toc = time.perf_counter()
        self.profiled_time['kdtree'] = toc - tic
        
        # Compute normalized direction for each edge.
        tic = time.perf_counter()
        directions = compute_directions(self.positions, self.edges)
        toc = time.perf_counter()
        self.profiled_time['direction'] = toc - tic
        
        # Compute view stacks for each edge.
        tic = time.perf_counter()
        self.view_from_stack, self.view_to_stack = compute_view_stacks(
            self.lct_directions, self.edges, directions
        )
        toc = time.perf_counter()
        self.profiled_time['view_stacks'] = toc - tic
    
    def config_l_mask(self, seed=0):
        rng = np.random.default_rng(seed)
        self.lct_mask = np.zeros((self.n_sat, self.N_LCT_PER_SAT), dtype=np.float32)
        self.lct_mask[:,0] = 1.0
        self.lct_mask[:,1] = 1.0
        # self.lct_mask[:,2] = 1.0
        # self.lct_mask[:,3] = 1.0
        
        # self.lct_mask = khot_matrix(self.n_sat, self.N_LCT_PER_SAT, 2, rng=rng, dtype=np.float32)
        # print(self.lct_mask.sum(axis=1), self.lct_mask.shape)
        
        # p= 0.5
        # n_lct4_sat = int(self.n_sat * p)
        # permuted_indices = rng.permutation(np.arange(0, self.n_sat))
        # lct4_indices = permuted_indices[:n_lct4_sat]
        # self.lct_mask[lct4_indices, :] = np.ones((n_lct4_sat, self.N_LCT_PER_SAT), dtype=np.float32) 
        
        
        # p = 0.4
        # self.lct_mask = rng.choice([0, 1], size=(self.n_sat, self.N_LCT_PER_SAT), p=[1-p, p]).astype(np.float32)
        # self.lct_mask += khot_matrix(self.n_sat, self.N_LCT_PER_SAT, 1, rng=rng, dtype=np.float32)
        # self.lct_mask[self.lct_mask> 1] = 1  # Ensure no values exceed 1.


        # n_lct2_sat = int(self.n_sat * lct2_rho)
        # n_lct4_sat = int(self.n_sat * lct4_rho)        
        # permuted_indices = rng.permutation(np.arange(0, self.n_sat))
        # self.lct2_indices = permuted_indices[:n_lct2_sat]
        # self.lct4_indices = permuted_indices[n_lct2_sat:n_lct2_sat + n_lct4_sat]
        # self.lct_mask = khot_matrix(self.n_sat, self.N_LCT_PER_SAT, 1, rng=rng, dtype=np.float32)
        # self.lct_mask[self.lct2_indices] = khot_matrix(n_lct2_sat, self.N_LCT_PER_SAT, 2, rng=rng, dtype=np.float32)
        # self.lct_mask[self.lct4_indices] = khot_matrix(n_lct4_sat, self.N_LCT_PER_SAT, 4, rng=rng, dtype=np.float32)
    
    def _apply_lt_mask(self):
        self._print(f'Updating LCT mask... {self.update_count}')
        if self.lct_mask is not None:
            self.view_from_stack, self.view_to_stack = apply_mask_to_view(self.lct_mask, self.edges, self.view_from_stack, self.view_to_stack)

    def _get_view_cos(self):
        self._print(f'Computing view cosines... {self.update_count}')
        direction = compute_directions(self.positions, self.filtered_sat_pair_repeated)
        ret = compute_lt_cos(self.lct_directions, self.filtered_sat_pair_repeated, self.filtered_lct_pair_expanded, direction, self.N_LCT_PER_SAT)
        return ret        
    
    def _update_p_lisl(self):
        self._print(f'Updating potential LISL... {self.update_count}')
        """Update potential LISL visualizations."""
        # Generate boolean indicators using the precomputed threshold.
        tic = time.perf_counter()
        self.filtered_edges, filtered_view_from_stack, filtered_view_to_stack, p_lisl_LT_pair = filter_and_compute_pair(
            self.edges, self.view_from_stack, self.view_to_stack, self.cos_threshold
        )
        toc = time.perf_counter()
        self.profiled_time['p_lisl_LT_pair'] = toc - tic
        
        # Expand edges and filter using the computed pair indicator.
        tic = time.perf_counter()
        # Only one direction of edges
        self.filtered_sat_pair_repeated, self.filtered_lct_pair_expanded = expand_and_filter_edges(self.filtered_edges, p_lisl_LT_pair)
        toc = time.perf_counter()
        self.profiled_time['expand_edges'] = toc - tic
        
        if np.asarray(self.filtered_lct_pair_expanded).size == 0:
            self._print("No potential LISL edges found.")
            return
        
        # Filter view stacks for valid edges and compute pairwise minimum.
        tic = time.perf_counter()
        self.view_LT_pair_min_cos = compute_view_LT_pair_min_cos(filtered_view_from_stack, filtered_view_to_stack, p_lisl_LT_pair.reshape(-1))
        toc = time.perf_counter()
        self.profiled_time['view_LT_pair'] = toc - tic
        
        tic = time.perf_counter()
        # Draw the potential LISL edges.
        if self.PLOT_POTENTIAL_LISL:
            # Build the color array using np.array for clarity.
            edges_color_data, p_lisl_data = optimize_edge_and_color_data(self.EDGE_COLOR, self.filtered_sat_pair_repeated, self.filtered_lct_pair_expanded, self.positions)
            edges_color_data[:, 3] = 0.25
            for viz in self.viz_list:
                # Update the potential LISL lines.
                viz['p_lisl'].set_data(pos=p_lisl_data, color=edges_color_data, width=self.PLOT_POTENTIAL_LISL_LINE_WIDTH, connect='segments')

        toc = time.perf_counter()
        self.profiled_time['draw_p_lisl'] = toc - tic
           
    def update_space(self):
        """
        inital update function called on each timer tick to update the simulation.
        """
        start_time = time.perf_counter()
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.Process().memory_info().rss / 1e6
        self._print(f"CPU Usage: {cpu_usage}%, Memory Usage: {mem_usage:.2f} MB")

        try:
            tic = time.perf_counter()
            self._update_earth()
            toc = time.perf_counter()
            self.profiled_time['earth_rotation'] = toc - tic
            
            tic = time.perf_counter()
            self._update_satellite_positions()
            toc = time.perf_counter()
            self.profiled_time['satellite_positions'] = toc - tic
            
            tic = time.perf_counter()
            self._update_satellite_arrows()
            toc = time.perf_counter()
            self.profiled_time['satellite_arrows'] = toc - tic
            
            tic = time.perf_counter()
            self._assign_lct()
            toc = time.perf_counter()
            self.profiled_time['assign_lct'] = toc - tic
                
            tic = time.perf_counter()
            self._update_p_satp()
            toc = time.perf_counter()
            self.profiled_time['_update_p_satp'] = toc - tic
            
            tic = time.perf_counter()
            self._apply_lt_mask()
            toc = time.perf_counter()
            self.profiled_time['_apply_lt_mask'] = toc - tic
            
            tic = time.perf_counter()
            self._update_p_lisl()
            toc = time.perf_counter()
            self.profiled_time['_update_p_lisl'] = toc - tic

            if self.PLOT_TEXT:        
                text = ""
                text += f"#n_sats: {self.n_sat}\n"
                text += f"#n_q_es: {self.edges.shape[0]}\n"
                text += f"#n_p_sp: {self.filtered_edges.shape[0]}\n"
                text += f"#n_p_lp: {self.filtered_lct_pair_expanded.shape[0]}\n"
                text += f"FOR_THETA: +/-{self.FOR_THETA_HALF:.0f}°\n"
                text += f"MAX_DIST: {self.LISL_MAX_DISTANCE:.0f} km\n"
                for viz in self.viz_list:
                    viz['text_top_left'].text = text
                
                text = ""
                text += f"AVG: {self.average_update_time:.4f} s\n"
                text += f"UPD: {self.update_count}\n"
                text += f"TOT: {time.perf_counter() - self.real_start_time:.2f} s\n"
                text += f"FPS: {1./self.average_update_time:.2f}\n"
                text += f"TSc: {self.TIME_SCALE:.2f}\n"
                text += f"SWT: {self.accumulated_update_time*self.TIME_SCALE:.2f} s\n"
                text += f"DAT: {self.get_simulation_time().utc_strftime('%Y-%m-%d %H:%M:%S')}\n"
                text += f"CPU: {psutil.cpu_percent()}%\n"
                text += f"MEM: {psutil.Process().memory_info().rss / 1e6:.2f} MB\n"
                for viz in self.viz_list:
                    viz['text_bot_left'].text = text
                    
                text = ""
                for key, value in self.profiled_time.items():
                    text += f"{key}: {value:.4f} s\n"
                
                for viz in self.viz_list:
                    viz['text_top_right'].text = text
                    
        except Exception as e:
            print("Unexpected error during update: %s" % e)
            # Stop the simulation
            app.quit()
            exit(1)
            
        elapsed = time.perf_counter() - start_time
        self.accumulated_update_time += elapsed
        self.average_update_time = 0.9 * self.average_update_time + 0.1 * elapsed

    def _update_o_lisl(self, satp = None, edge_weight=None, viz=None, binary=False):
        # Update optional LISL lines.
        if np.asarray(satp).size == 0:
            return
        tic = time.perf_counter()
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
            viz['o_lisl'].set_data(pos=o_lisl_data, color=edges_color_data, width=2, connect='segments')
        
        toc = time.perf_counter()
        self.profiled_time['draw_matching'] = toc - tic

    def update(self, pull=False):
        """
        Update the simulation at each timer tick.
        """
        self._print(f"Timer event triggered. Update count: {self.update_count}")
        self.update_count += 1
        try:
            if self.data_thread is not None and self.data_thread.is_alive():
                if time.perf_counter() - self.data_thread_last_tickT > 1:
                    self._printalltime(f"Data thread is still running... {self.update_count}, {self.N_STEP}")
                    self.data_thread_last_tickT = time.perf_counter()
            else:
                self._printalltime(f"Starting data thread... {self.update_count}, {self.N_STEP}")
                self.data_thread = Thread(target=self.step, args=(), daemon=True)
                self.data_thread.start()
                self.data_thread_start_time = time.perf_counter()
                self.data_thread_last_tickT = time.perf_counter()
                if pull:
                    self.data_thread.join()
        except Exception as e:
            print("Error during update: %s" % e)
            import traceback
            traceback.print_exc()
            # Stop the simulation
            app.quit()
            exit(1)
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.Process().memory_info().rss / 1e6
        self._print(f"CPU Usage: {cpu_usage}%, Memory Usage: {mem_usage:.2f} MB")    
  
    @counted
    def step(self):
        self.run_step()
    
    def run_step(self):
        pass
    
    def run(self, TOT_STEPS=1000, visualize=False):
        """
        Run the simulation.
        """
        self.TOT_STEPS = TOT_STEPS
        self.VISUALIZE = visualize
        if visualize:
            self.setup_visualization()
        # Initialize the constellation
        self.update_space()
        self._printalltime("Starting simulation...")
        while True:
            if visualize:
                self.visualize()
                self.set_viz_data()

            if self.N_STEP >= self.TOT_STEPS:
                if self.data_thread is not None and self.data_thread.is_alive():
                    self._printalltime("MAX iterataion reached, waiting for data thread to finish")
                    continue
                self._printalltime("Simulation finished")
                break
            else:
                self.update(pull= not visualize)
            
    def visualize(self):
        self.canvas.update()
        app.process_events()
        time.sleep(0.01)
        
    def set_viz_data(self):
        pass
    
    @property
    def EDGE_COLOR(self):
        return np.array([self.FRONT_COLOR, self.BACK_COLOR, self.RIGHT_COLOR, self.LEFT_COLOR], dtype=np.float32)