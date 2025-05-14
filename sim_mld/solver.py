from numba import njit

from sim_mld.dijkstra import *
from sim_mld.lisl_channel_model import *

import scipy.sparse as sp
import cvxpy as cp

from sim_src.util import STATS_OBJECT

@njit(cache=True)
def greedy_max_weight_matching(E: np.ndarray) -> list:
    """
    Compute a greedy heuristic maximum weight matching for a NumPy array of edges.
    
    Parameters:
        E (np.ndarray): Array with shape (num_edges, 3) where each row is [u, v, weight].
    
    Returns:
        list: List of tuples (u, v) representing the selected edges.
    """
    sorted_indices = np.argsort(-E[:, 2])
    E_sorted = E[sorted_indices]
    
    matching = []
    matched_nodes = set()
    
    for edge in E_sorted:
        u, v, weight = edge
        u, v = int(u), int(v)
        if u not in matched_nodes and v not in matched_nodes:
            matching.append((u, v))
            matched_nodes.add(u)
            matched_nodes.add(v)
    
    return matching

@njit(cache=True)
def graph_partition(num_nodes, indptr, indices):
    """
    Partition the graph into connected components and return an array 
    indicating each node's component id. If the graph is fully connected,
    all entries are 0.
    
    Parameters:
      num_nodes : int
          The total number of nodes.
      indptr : np.ndarray of shape (num_nodes+1,) of int64
          CSR pointer array.
      indices : np.ndarray of int64
          CSR indices array (neighbors of each node).
    
    Returns:
      comp : np.ndarray of shape (num_nodes,) of int64
          Array where comp[i] is the component id of node i.
          All nodes in a connected component share the same id.
    """
    # Initialize each node as not visited (-1 indicates unassigned)
    comp = -np.ones(num_nodes, dtype=np.int64)
    comp_id = 0  # Component identifier
    # Allocate an array for DFS; maximum size equals num_nodes.
    stack = np.empty(num_nodes, dtype=np.int64)
    
    # Process each node: if it is unvisited, perform DFS to mark the component.
    for i in range(num_nodes):
        if comp[i] == -1:
            # Start DFS from node i.
            stack_top = 0
            stack[stack_top] = i
            stack_top += 1
            comp[i] = comp_id
            while stack_top > 0:
                stack_top -= 1
                node = stack[stack_top]
                # Traverse the neighbors of 'node'
                for j in range(indptr[node], indptr[node + 1]):
                    neighbor = indices[j]
                    if comp[neighbor] == -1:
                        comp[neighbor] = comp_id
                        stack[stack_top] = neighbor
                        stack_top += 1
            comp_id += 1  # Move to next component id for a new DFS run
    return comp

@njit(cache=True)
def random_source_target_pairs(N, P):
    """
    Generate P random (source, target) pairs for a graph with N nodes,
    ensuring no self loops (source != target).
    
    Parameters:
      N : int
          Number of nodes (assumed to be labeled 0 to N-1).
      P : int
          Number of pairs to generate.
          
    Returns:
      sources : np.ndarray of shape (P,)
          Array of source nodes.
      targets : np.ndarray of shape (P,)
          Array of corresponding target nodes.
    """
    # Initially generate P random source and target values.
    sources = np.random.randint(0, N, P)
    targets = np.random.randint(0, N, P)
    
    # For indices where source equals target, re-sample the target.
    # This loop continues until no self-loops remain.
    while np.any(sources == targets):
        mask = (sources == targets)
        targets[mask] = np.random.randint(0, N, np.sum(mask))
        
    return sources, targets

class price_graph:
    def __init__(self, n_sat, possible_sat_pair_expanded, initial_prices=1.):
        self.n_sat = n_sat
        # Initialize edge prices
        csr_pair = build_csr(self.n_sat, possible_sat_pair_expanded, np.ones(possible_sat_pair_expanded.shape[0])*initial_prices)
        self.price_graph = sp.csr_matrix((csr_pair[2], csr_pair[1], csr_pair[0]), shape=(self.n_sat, self.n_sat))
    
    def get_prices(self, possible_sat_pair_expanded):
        # Compute edge price based on the price graph
        edge_price_on_graph_idx_wgt = csr_to_edge_list(self.price_graph.indptr, self.price_graph.indices, self.price_graph.data)
        prices = extract_weights(possible_sat_pair_expanded, edge_price_on_graph_idx_wgt)
        # Check if np.nan is in the weights
        if np.isnan(prices).any():
            raise ValueError("NaN found in edge weights")
        
        # Check if the weights are all positive
        if np.any(prices < 0):
            raise ValueError("Negative weights found in edge weights")
    
        return prices
    
    def add_prices(self, d_price_graph):
        self.price_graph += d_price_graph
        self.price_graph.data[self.price_graph.data < 0] = 0

class mr_solver(STATS_OBJECT):
    WAVELENGTH = 1.55e-6  # Wavelength in meters (1.55 microns typical in telecom)
    ANGULAR_SPREADING = 100e-6  # Angular spreading in radians
    BEAM_WAIST = w0_from_angular_spreading(ANGULAR_SPREADING, WAVELENGTH)  # Beam waist in meters
    PEAK_POWER_W = 20  # Convert dBm to Watts
    BANDWIDTH = 1e9  # 1 GHz bandwidth
    RESPONSIVITY = 0.5  # 0.8 A/W responsivity
    APERTURE_AREA = 1e-2  # Area in m^2 (example)
    NOISE_CURRENT = 3e-7  # Example noise in A rms
    JITTER = 10e-6  # Jitter in radians
    EPSILON = 1e-3  # Epsilon for relaxed capacity calculations

    EARTH_RADIUS = 6371e3  # Earth radius in meters
    N_LCT_PER_SAT = 4  # Number of LCTs per satellite
    
    MIN_CAPACITY = 1
    
    INIT_PRICES = 0.
    
    def __init__(self):
        self.ALPHA = 0.1
        
        self.objective_mode = "maxlog"
        
        self.n_sat = 0
        self.positions = None
        self.possible_sat_pair_expanded = None
        self.possible_lct_pair_expanded = None 
        
        self.possible_lct_pair_edge_capacity = None
        
        self.data_source = None
        self.data_target = None
        self.s_t_data_rate = None
        
        self.price_graph: price_graph = None
                
    def init_constellation(self, possible_sat_pair_expanded, possible_lct_pair_expanded, positions):
        self.possible_sat_pair_expanded = possible_sat_pair_expanded
        self.possible_lct_pair_expanded = possible_lct_pair_expanded
        
        capacity = self.compute_capacity(
            np.linalg.norm(positions[self.possible_sat_pair_expanded[:, 0]] - positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
        )
        self.possible_sat_pair_expanded = self.possible_sat_pair_expanded[capacity > self.MIN_CAPACITY]
        self.possible_lct_pair_expanded = self.possible_lct_pair_expanded[capacity > self.MIN_CAPACITY]
        
        # Initialize edge capacities
        num_edges = len(possible_lct_pair_expanded)
        self.possible_lct_pair_edge_capacity = np.zeros(num_edges)
        
        # Set initial positions
        self.positions = positions
        self.n_sat = positions.shape[0]
        
        self.price_graph = price_graph(self.n_sat, self.possible_sat_pair_expanded, self.INIT_PRICES)
    
        self.update_source_target_pairs()
    
    @classmethod
    def compute_capacity(cls, distance):
        distance = distance * cls.EARTH_RADIUS
        return capacity_relaxed(
            cls.BANDWIDTH, cls.RESPONSIVITY, cls.APERTURE_AREA, cls.NOISE_CURRENT,
            cls.PEAK_POWER_W, cls.BEAM_WAIST, cls.WAVELENGTH, distance,
            cls.JITTER, cls.EPSILON
        )
        
    def check_connected(self):
        # Check if the graph is connected
        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, np.ones(self.possible_sat_pair_expanded.shape[0]))
        comp = graph_partition(self.n_sat, indptr, indices)
        num_components = np.unique(comp).shape[0]
        self._print(f"Number of components: {num_components}")
        return num_components == 1, comp

    def update_source_target_pairs(self, n_pair=1, data_rate=0., seed=0):
        rng = np.random.default_rng(seed)
        
        sources = rng.integers(0, self.n_sat, n_pair)
        targets = rng.integers(0, self.n_sat, n_pair)
        while np.any(sources == targets):
            mask = (sources == targets)
            targets[mask] = rng.integers(0, self.n_sat, np.sum(mask))
        
        self.data_source, self.data_target = sources, targets
        self._print(f"Source: {self.data_source}, Target: {self.data_target}")
        self.s_t_data_rate = np.ones(self.data_source.shape[0]) * data_rate
        
    def get_dual_matching(self):
        self._print("Computing dual matching")
        prices = self.price_graph.get_prices(self.possible_sat_pair_expanded)
        capacity = self.compute_capacity(
            np.linalg.norm(self.positions[self.possible_sat_pair_expanded[:, 0]] - self.positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
        )
        
        edge_weights_pr = prices * capacity
        self._print(f"Ec: max: {np.max(capacity)}, min: {np.min(capacity)}")
        self._print(f"Mw: max: {np.max(edge_weights_pr)}, min: {np.min(edge_weights_pr)}")
        weighted_edges = np.column_stack((self.possible_lct_pair_expanded, edge_weights_pr))
        matching = greedy_max_weight_matching(weighted_edges)
        m = len(matching)
        flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
                                  for x in pair), dtype=int, count=2*m)
        connected_lct = flat_array.reshape(-1, 2)
        connected_sat = connected_lct // self.N_LCT_PER_SAT
        self._print("Computing dual done")
        return connected_sat, connected_lct
    
    def get_dual_srouting(self, debug=False):
        self._print("Computing dual srouting")
        prices = self.price_graph.get_prices(self.possible_sat_pair_expanded)
        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, prices)
        self._print("build csr done")
        # costs, lengths, paths_all = multi_dijkstra_with_paths_aw_b(self.n_sat, indptr, indices, self.data_source, self.data_target, data, self.s_t_data_rate, np.ones_like(self.s_t_data_rate))
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        self._print("Computing dual srouting done")

        if debug:
            for i in range(self.data_source.shape[0]):
                if costs[i] < 1e12:
                    path = paths_all[i, :lengths[i]]
                    print("Query {}: from {} to {}: cost = {}, path = {}".format(
                        i, self.data_source[i], self.data_target[i], costs[i], path))
                else:
                    print("Query {}: from {} to {} is unreachable.".format(
                        i, self.data_source[i], self.data_target[i]))
        # if np.any(lengths == 0):
            # raise ValueError("No valid lengths found, The graph is not connected")
        return costs, lengths, paths_all

    def get_dual_objective(self, with_prices=False):
        self._print("Computing dual objective")
        connected_sat, connected_lct = self.get_dual_matching()
        costs, lengths, paths_all = self.get_dual_srouting()

        # Compute the edge capacity for the connected satellites
        capacity_on_graph = self.compute_capacity(
            np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
        )
        prices = self.price_graph.get_prices(connected_sat)
        if costs.min() < 1:
            lambda_times_capacity = (-1+ costs[costs < 1]).sum()*10000 - np.sum(prices * capacity_on_graph)
        else:
            lambda_times_capacity = -np.sum(prices * capacity_on_graph)
        if with_prices:
            return lambda_times_capacity, np.concatenate((self.possible_sat_pair_expanded,self.price_graph.get_prices(self.possible_sat_pair_expanded).reshape(-1,1)), axis=1)
        else:
            return lambda_times_capacity

    def get_prim_objective(self, with_rates=False):
        self._print("Computing prim objective")
        connected_sat, connected_lct = self.get_dual_matching()
        prices = self.price_graph.get_prices(connected_sat)
        indptr, indices, data = build_csr(self.n_sat, connected_sat, prices)
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )
        
        rates = self.get_rates(srouting)
        self._print(f"Real rate: MAX: {np.max(rates)}, MIN: {np.min(rates)}")
        self._print(f"Appr rate: MAX: {np.max(self.s_t_data_rate)}, MIN: {np.min(self.s_t_data_rate)}")
        if not with_rates:
            return -np.sum(rates)
        else:
            return -np.sum(rates), rates, srouting, (costs, lengths, paths_all)
      
    def get_rates(self, srouting, matching, mode="maxsum"):
        # Compute the edge capacity for the connected satellites
        connected_sat = matching // self.N_LCT_PER_SAT
        capacity = self.compute_capacity(
            np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
        )
        indptr, indices, data = build_csr(self.n_sat, connected_sat, capacity, merging_method="sum")
        edge_capacity = csr_to_edge_list(indptr, indices, data)
        
        
        #TODO: Handle the case when there are multiple LISL connections between the same pair of satellites
        self._print("Computing max rate")
        if np.asarray(srouting).size == 0:
            return np.zeros(self.data_source.shape[0])
        # edges: K×4 numpy array, columns = [u, v, s, t]
        flow_pairs = srouting[:, 2:4]
        unique_pairs_view, flow_ids = np.unique(flow_pairs, return_inverse=True, axis=0)
        F = unique_pairs_view.shape[0]

        uv_view = srouting[:, :2]                                 # the (u,v) pairs
        uniq_uv_view, edge_ids = np.unique(uv_view, return_inverse=True, axis=0)
        E = uniq_uv_view.shape[0]

        P = sp.coo_matrix(
            (np.ones(srouting.shape[0]), (edge_ids, flow_ids)),
            shape=(E, F)
        ).tocsr()
        
        # --- Capacity constraints ------------------------------------------------
        uv = uniq_uv_view.reshape(-1, 2).astype(np.int64)
        capacity = extract_weights(uv, edge_capacity)

        # --- LP solve -------------------------------------------------------------
        r = cp.Variable(F, nonneg=True)
        if mode == "maxmin":
            prob = cp.Problem(cp.Maximize(cp.min(r)), [P @ r <= capacity])
        elif mode == "maxlog":
            prob = cp.Problem(cp.Maximize(cp.sum(cp.log(r+1))), [P @ r <= capacity])
        else:
            prob = cp.Problem(cp.Maximize(cp.sum(r)), [P @ r <= capacity])
        prob.solve(solver=cp.SCS)

        opt = r.value
        
        rate_map = {
            (int(src), int(tgt)): float(rate)
            for (src, tgt), rate in zip(unique_pairs_view, opt)
        }
        # 2. For each original pair, look it up (default to 0.0 if missing)
        rates = np.array([
            rate_map.get((src, tgt), 0.0)
            for src, tgt in zip(self.data_source, self.data_target)
        ])
        return rates
        
    def update_step_rates_prices(self):
        pass


if __name__ == '__main__':
    print(np.exp(-np.inf))