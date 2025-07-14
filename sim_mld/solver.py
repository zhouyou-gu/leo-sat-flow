from numba import njit
import pandas as pd

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

def prepare_group_index(targets):
    """
    Map each unique target to a group index 0..G-1.
    Returns:
      uniq_t   : int64[G] array of sorted unique targets
      t_inv    : int32[P] array so that t_inv[i] is group index of targets[i]
    """
    uniq_t, inv = np.unique(targets, return_inverse=True)
    return uniq_t.astype(np.int64), inv.astype(np.int32)

@njit(parallel=True, cache=True)
def _topk_numba(sources, t_inv, costs, k):
    """
    Core Numba‐parallel routine.
    Args:
      sources : int64[P]
      t_inv   : int32[P]  # inverse map into 0..G-1
      costs   : float64[P]
      k       : int32
    Returns:
      out     : int64[G, k, 2]  sentinel‐padded
    """
    P = sources.shape[0]
    G = t_inv.max() + 1

    # sentinel pad = -1
    out = np.full((G, k, 2), -1, dtype=np.int64)

    # for each target‐group in parallel
    for g in prange(G):
        # count members in this group
        cnt = 0
        for i in range(P):
            if t_inv[i] == g:
                cnt += 1

        if cnt == 0:
            # skip entirely, leave sentinel
            continue

        # gather indices
        idxs = np.empty(cnt, np.int32)
        pos = 0
        for i in range(P):
            if t_inv[i] == g:
                idxs[pos] = i
                pos += 1

        # how many to take
        m = cnt if cnt < k else k

        # find the m smallest costs via argpartition
        # note: m-1 is safe even if m=1
        sel = np.argpartition(costs[idxs], m-1)[:m]

        # write them out
        for j in range(m):
            ii = idxs[sel[j]]
            out[g, j, 0] = sources[ii]
            out[g, j, 1] = g     # store group index here

    return out

@njit(parallel=True, cache=True)
def _replace_group_indices(raw, uniq_t):
    """
    Replace group indices in the raw output with real target values.
    
    Args:
      raw     : int64[G, k, 2]  sentinel‐padded
      uniq_t  : int64[G]        unique target values
    Returns:
      raw     : int64[G, k, 2]  with group indices replaced by real target values
    """
    G, K, _ = raw.shape

    for g in prange(G):
        for j in range(K):
            if raw[g, j, 0] != -1:
                # replace group index with real target value
                raw[g, j, 1] = uniq_t[g]
    return raw

def top_k_pairs_to_t_with_min_cost_numba(sources, targets, costs, k=5):
    """
    Wrapper that:
      1) builds group index
      2) calls the parallel Numba function
      3) maps group indices back to real target values
      4) flattens and drops sentinels
    """
    # 1) map targets → 0..G-1
    uniq_t, t_inv = prepare_group_index(targets)

    # 2) call compiled routine
    raw = _topk_numba(sources, t_inv, costs, np.int32(k))

    # 3) replace group‐indices with real target values
    G, K, _ = raw.shape
    raw = _replace_group_indices(raw, uniq_t)
    # 4) flatten and filter out sentinel rows
    flat = raw.reshape(-1, 2)
    mask = (flat[:, 0] != -1)  # keep only real rows
    return flat[mask]
    


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
    RESPONSIVITY = 0.5  # A/W responsivity
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
        
        # solver states
        self.objective_mode = "maxsum"
        self.price_graph: price_graph = None

        self.n_sat = 0
        self.positions = None
        self.possible_sat_pair_expanded = None
        self.possible_lct_pair_expanded = None 
        
        # traffic info
        self.data_source = None
        self.data_target = None
     
        self.s_t_traffic_rates = None
     
        self.forward_traffic_capacity = None
        self.forward_traffic_demand = None   
    
    def _reset_price_graph(self, initial_prices=INIT_PRICES):
        self.price_graph = price_graph(self.n_sat, self.possible_sat_pair_expanded, initial_prices)
    
    def _remove_non_connected_s_t_pairs(self):
        """
        Remove pairs that are not connected in the graph.
        """
        satellite_distances = np.linalg.norm(self.positions[self.possible_sat_pair_expanded[:, 0]] - self.positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
        satellite_distances = satellite_distances * self.EARTH_RADIUS  # Convert to meters
        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, satellite_distances)
        tic = self._get_tic()
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        tim = self._get_tim(tic)
        self._printalltime(f"Multi-Dijkstra took {tim:.2f} us")
        self.data_source = self.data_source[lengths > 0]
        self.data_target = self.data_target[lengths > 0]
        costs = costs[lengths > 0]
        self.s_t_traffic_rates = self.s_t_traffic_rates[lengths > 0]
        
        ## Find top-k pairs with minimum costs
        tic = self._get_tic()
        # s_t = top_k_pairs_to_t_with_min_cost_numba (
        #     self.data_source, self.data_target, costs, k=5
        # )
        pairs = np.column_stack((self.data_source, self.data_target))
        df = pd.DataFrame(pairs, columns=['s', 't'])
        df['cost'] = costs

        # For each t, take the K rows with smallest cost
        topk = (
            df
            .groupby('t', group_keys=False)[['s', 't','cost']]       # group by destination t
            .apply(lambda g: g.nsmallest(5, 'cost'))
            # .apply(lambda g: g.sample(5,replace=False))
            .reset_index(drop=True)
        )
        s_t = topk[['s', 't']].values
        tim = self._get_tim(tic)
        self._printalltime(f"Top-k pairs took {tim:.2f} us")




        pair_rate_list = np.column_stack((self.data_source, self.data_target, self.s_t_traffic_rates))
        filtered_rates = extract_weights(s_t, pair_rate_list, none_value=-100.0)
        self.data_source = s_t[:, 0]
        self.data_target = s_t[:, 1]
        self.s_t_traffic_rates = filtered_rates
        if np.any(self.s_t_traffic_rates < 0):
            raise ValueError("Negative rates found in s_t_traffic_rates")

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
        
        rates = self.get_rates_dual(costs)
        
        ret = - np.sum(prices * capacity_on_graph)
        ret += np.sum(rates*(-1+ costs))
        if with_prices:
            return ret, np.concatenate((self.possible_sat_pair_expanded,self.price_graph.get_prices(self.possible_sat_pair_expanded).reshape(-1,1)), axis=1)
        else:
            return ret

    def get_prim_objective(self, with_rates=False):
        self._print("Computing prim objective")
        connected_sat, connected_lct = self.get_dual_matching()

        prices = self.price_graph.get_prices(connected_sat)
        indptr, indices, data = build_csr(self.n_sat, connected_sat, prices)
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )

        rates = self.get_rates_prim(srouting, connected_lct, mode=self.objective_mode)
        self._print(f"Real rate: MAX: {np.max(rates)}, MIN: {np.min(rates)}")
        self._print(f"Appr rate: MAX: {np.max(self.s_t_traffic_rates)}, MIN: {np.min(self.s_t_traffic_rates)}")
        if not with_rates:
            return -np.sum(rates)
        else:
            return -np.sum(rates), rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct


    def get_prim_objective_mwm(self, with_rates=False):
        capacity = self.compute_capacity(
            np.linalg.norm(self.positions[self.possible_sat_pair_expanded[:, 0]] - self.positions[self.possible_sat_pair_expanded[:, 1]], axis=1)
        )        
        edge_weights_pr = capacity
        weighted_edges = np.column_stack((self.possible_lct_pair_expanded, edge_weights_pr))
        matching = greedy_max_weight_matching(weighted_edges)
        m = len(matching)
        flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
                                  for x in pair), dtype=int, count=2*m)
        connected_lct = flat_array.reshape(-1, 2)
        connected_sat = connected_lct // self.N_LCT_PER_SAT
        distance = np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
        capacity = self.compute_capacity(distance)

        indptr, indices, data = build_csr(self.n_sat, connected_sat, 1/capacity)
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )
        
        rates = self.get_rates_prim(srouting, connected_lct, mode=self.objective_mode)
        self._print(f"Real rate: MAX: {np.max(rates)}, MIN: {np.min(rates)}")
        if not with_rates:
            return -np.sum(rates)
        else:
            return -np.sum(rates), rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct

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
        prob.solve(solver=cp.HiGHS)

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
    
    def get_rates_prim(self, srouting, matching, mode="maxsum"):
        # Compute the edge capacity for the connected satellites
        connected_sat = matching // self.N_LCT_PER_SAT
        capacity = self.compute_capacity(
            np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
        )
        indptr, indices, data = build_csr(self.n_sat, connected_sat, capacity, merging_method="sum")
        edge_capacity = csr_to_edge_list(indptr, indices, data)
        
        if np.asarray(srouting).size == 0:
            return np.zeros(self.data_source.shape[0])
        
        flow_pairs = srouting[:, 2:4]
        unique_pairs_view, flow_ids = np.unique(flow_pairs, return_inverse=True, axis=0)
        F = unique_pairs_view.shape[0]
        
        uv_view = srouting[:, :2]                                 # the (u,v) pairs
        uniq_uv_view, edge_ids = np.unique(uv_view, return_inverse=True, axis=0)
        E = uniq_uv_view.shape[0]

        # Create the flow matrix
        P = sp.coo_matrix(
            (np.ones(srouting.shape[0]), (edge_ids, flow_ids)),
            shape=(E, F)
        ).tocsr()
        
        # --- Capacity constraints ------------------------------------------------
        uv = uniq_uv_view.reshape(-1, 2).astype(np.int64)
        capacity = extract_weights(uv, edge_capacity)
        
        # Initialize the variable for rates
        r = cp.Variable(F, nonneg=True)
        
        # Source and target rate constraints
        St = sp.coo_matrix((np.ones(F), 
                          (unique_pairs_view[:, 0], np.arange(F))),
                          shape=(self.n_sat, F)).tocsr()

        Dt = sp.coo_matrix((np.ones(F),
                          (unique_pairs_view[:, 1], np.arange(F))),
                          shape=(self.n_sat, F)).tocsr()
        
        constraints = []
        # Source rate bound constraints
        constraints.append(St @ r <= self.forward_traffic_capacity)
        # Target rate bound constraints
        constraints.append(Dt @ r <= self.forward_traffic_demand)

        # Flow conservation constraints
        constraints.append(P @ r <= capacity)
        
        # --- LP solve -------------------------------------------------------------
        prob = cp.Problem(cp.Maximize(cp.sum(r)), constraints)
        prob.solve(solver=cp.HIGHS)

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
        if np.isnan(rates).any():
            raise ValueError("NaN found in rates")
        if np.any(rates < 0):
            raise ValueError("Negative rates found in rates")
        
        self._print(f"Real rate: MAX: {np.max(rates)}, MIN: {np.min(rates)}")
        return rates

    def get_rates_dual(self, costs):
        F = self.data_source.shape[0]
        # Initialize the variable for rates
        r = cp.Variable(F, nonneg=True)
        
        # Source and target rate constraints
        St = sp.coo_matrix((np.ones(F), 
                          (self.data_source, np.arange(F))),
                          shape=(self.n_sat, F)).tocsr()

        Dt = sp.coo_matrix((np.ones(F),
                          (self.data_target, np.arange(F))),
                          shape=(self.n_sat, F)).tocsr()

        constraints = []
        # Source rate bound constraints
        constraints.append(St @ r <= self.forward_traffic_capacity)
        # Target rate bound constraints
        constraints.append(Dt @ r <= self.forward_traffic_demand)
        
        # --- LP solve -------------------------------------------------------------
        obj = cp.sum(cp.multiply(-costs+1, r))
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(solver=cp.HIGHS)
        
        opt = r.value
        return opt


    def update_step_rates_prices(self):
        pass

if __name__ == '__main__':
    print(np.exp(-np.inf))