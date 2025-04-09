from numba import njit

from sim_alg.dijkstra import build_csr, multi_dijkstra_with_paths
from sim_alg.lisl_channel_model import *


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


class mr_solver:
    WAVELENGTH = 1.55e-6  # Wavelength in meters (1.55 microns typical in telecom)
    ANGULAR_SPREADING = 100e-6  # Angular spreading in radians
    BEAM_WAIST = w0_from_angular_spreading(ANGULAR_SPREADING, WAVELENGTH)  # Beam waist in meters
    PEAK_POWER_W = 10  # Convert dBm to Watts
    BANDWIDTH = 1e9  # 1 GHz bandwidth
    RESPONSIVITY = 0.5  # 0.8 A/W responsivity
    APERTURE_AREA = 1e-2  # Area in m^2 (example)
    NOISE_CURRENT = 3e-7  # Example noise in A rms
    JITTER = 10e-6  # Jitter in radians
    EPSILON = 1e-5  # Epsilon for relaxed capacity calculations

    EARTH_RADIUS = 6371e3  # Earth radius in meters
    def __init__(self):
        self.n_sat = 0
        self.positions = None
        self.possible_sat_pair_expanded = None
        self.possible_lct_pair_expanded = None 
        
        self.possible_lct_pair_edge_prices = None
        self.possible_lct_pair_edge_capacity = None
        
        self.s_t_data_rate = 1.
    
    def init_edges(self, possible_sat_pair_expanded, possible_lct_pair_expanded,positions):
        self.possible_sat_pair_expanded = possible_sat_pair_expanded
        self.possible_lct_pair_expanded = possible_lct_pair_expanded
        
        # Initialize edge prices and capacities
        num_edges = len(possible_lct_pair_expanded)
        self.possible_lct_pair_edge_prices = np.ones(num_edges)
        self.possible_lct_pair_edge_capacity = np.zeros(num_edges)
        
        # Set initial positions
        self.positions = positions
        self.n_sat = positions.shape[0]
        self.update_edge_capacity()
    
    @classmethod
    def compute_capacity(cls, distance):
        distance = distance * cls.EARTH_RADIUS
        return capacity_relaxed(
            cls.BANDWIDTH, cls.RESPONSIVITY, cls.APERTURE_AREA, cls.NOISE_CURRENT,
            cls.PEAK_POWER_W, cls.BEAM_WAIST, cls.WAVELENGTH, distance,
            cls.JITTER, cls.EPSILON
        )
    
    def update_edge_capacity(self):
        distance = np.linalg.norm(self.positions[self.possible_sat_pair_expanded[:, 0]] - self.positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
        self.edge_capacity = self.compute_capacity(distance)
        
    def get_dual_matching(self):
        edge_weights =  self.possible_lct_pair_edge_prices * self.possible_lct_pair_edge_capacity
        weighted_edges = np.column_stack((self.possible_lct_pair_expanded, edge_weights))
        matching = greedy_max_weight_matching(weighted_edges)
        m = len(matching)
        flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
                                  for x in pair), dtype=int, count=2*m)
        connected_lts = flat_array.reshape(-1, 2)
        connected_sat = connected_lts // 4
        
        
        return connected_sat, connected_lts
    
    def get_dual_srouting(self, debug=False):
        edge_weights = self.possible_lct_pair_edge_prices * self.s_t_data_rate + 1.
        s_t_source, s_t_target = random_source_target_pairs(self.n_sat, 1000)
        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, edge_weights)
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, data, s_t_source, s_t_target)
        
        if debug:
            for i in range(s_t_source.shape[0]):
                if costs[i] < 1e12:
                    path = paths_all[i, :lengths[i]]
                    print("Query {}: from {} to {}: cost = {}, path = {}".format(
                        i, s_t_source[i], s_t_target[i], costs[i], path))
                else:
                    print("Query {}: from {} to {} is unreachable.".format(
                        i, s_t_source[i], s_t_target[i]))
        
                    
    def check_connected(self):
        # Check if the graph is connected
        num_nodes = self.n_sat
        indptr, indices, data = build_csr(num_nodes, self.possible_sat_pair_expanded, self.possible_lct_pair_edge_prices)
        comp = graph_partition(num_nodes, indptr, indices)
        num_components = np.unique(comp).shape[0]
        return num_components == 1, comp
    
    

