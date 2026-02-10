from numba import njit
import pandas as pd

from sim_mld.dijkstra import *
from sim_mld.lisl_channel_model import *
from sim_mld.constants import (
    INFINITY_THRESHOLD,
    OPTICAL_RESPONSIVITY,
    OPTICAL_EFFICIENCY_BETA,
    DEFAULT_LCT_COUNT,
    EARTH_RADIUS_KM,
)
from sim_mld.capacity_calculator import CapacityCalculator
from sim_mld.matching_strategy import greedy_max_weight_matching, GreedyMatchingStrategy
from sim_mld.routing_strategy import DijkstraRoutingStrategy

import scipy.sparse as sp
import cvxpy as cp

from sim_src.util import STATS_OBJECT

# Note: greedy_max_weight_matching is imported from matching_strategy module

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
    
    def get_prices(self, possible_sat_pair_expanded, sum_both_direction=False):
        # Compute edge price based on the price graph
        edge_price_on_graph_idx_wgt = csr_to_edge_list(self.price_graph.indptr, self.price_graph.indices, self.price_graph.data)
        prices = extract_weights(possible_sat_pair_expanded, edge_price_on_graph_idx_wgt)
        if sum_both_direction:
            prices_ = extract_weights(possible_sat_pair_expanded[:, ::-1], edge_price_on_graph_idx_wgt)
            prices += prices_
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
    """
    Multi-resource solver for satellite network optimization.
    
    This class orchestrates matching and routing strategies to optimize
    traffic flows in satellite networks.
    """
    
    # Use CapacityCalculator for all capacity-related constants and computations
    N_LCT_PER_SAT = DEFAULT_LCT_COUNT  # Number of LCTs per satellite
    INIT_PRICES = 0.
    BETA = OPTICAL_EFFICIENCY_BETA
    
    def __init__(self):
        self.ALPHA = 0.1
        
        # Initialize strategy objects
        self.capacity_calculator = CapacityCalculator()
        self.matching_strategy = GreedyMatchingStrategy()
        self.routing_strategy = DijkstraRoutingStrategy()
        
        # solver states
        self.objective_mode = "maxsum"
        self.price_graph: price_graph = None

        self.n_sat = 0
        self.positions = None
        self.possible_sat_pair_expanded = None
        self.possible_lct_pair_expanded = None
        self.possible_lct_pair_expanded_view_cos = None

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
        K_VALUE = 5
        topk = (
            df
            .groupby('t', group_keys=False)[['s', 't','cost']]       # group by destination t
            .apply(lambda g: g.nsmallest(K_VALUE, 'cost'))
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
        """
        Compute link capacity based on distance.
        
        This method delegates to CapacityCalculator for consistency.
        
        Parameters
        ----------
        distance : float or np.ndarray
            Distance between satellites (normalized units)
            
        Returns
        -------
        float or np.ndarray
            Link capacity in Gbps
        """
        # Convert distance to meters and use capacity calculator
        distance_m = distance * CapacityCalculator.EARTH_RADIUS
        return CapacityCalculator.compute_capacity(distance_m)
        
    def check_connected(self):
        # Check if the graph is connected
        indptr, indices, data = build_csr(self.n_sat, self.possible_sat_pair_expanded, np.ones(self.possible_sat_pair_expanded.shape[0]))
        comp = graph_partition(self.n_sat, indptr, indices)
        num_components = np.unique(comp).shape[0]
        self._print(f"Number of components: {num_components}")
        return num_components == 1, comp

    def get_dual_matching(self):
        self._print("Computing dual matching")
        prices = self.price_graph.get_prices(self.possible_sat_pair_expanded, sum_both_direction=True)
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
        edge_in_both_direction = np.concatenate((self.possible_sat_pair_expanded, self.possible_sat_pair_expanded[:, ::-1]), axis=0)
        prices = self.price_graph.get_prices(edge_in_both_direction)
        indptr, indices, data = build_csr(self.n_sat, edge_in_both_direction, prices, sym_half=False)
        self._print("build csr done")
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        self._print("Computing dual srouting done")

        if debug:
            for i in range(self.data_source.shape[0]):
                if costs[i] < INFINITY_THRESHOLD:
                    path = paths_all[i, :lengths[i]]
                    print("Query {}: from {} to {}: cost = {}, path = {}".format(
                        i, self.data_source[i], self.data_target[i], costs[i], path))
                else:
                    print("Query {}: from {} to {} is unreachable.".format(
                        i, self.data_source[i], self.data_target[i]))
        # if np.any(lengths == 0):
            # raise ValueError("No valid lengths found, The graph is not connected")
        return costs, lengths, paths_all

    def get_dual_objective(self, with_prices=False, with_objective=False):
        self._print("Computing dual objective")
        connected_sat, connected_lct = self.get_dual_matching()
        costs, lengths, paths_all = self.get_dual_srouting()

        # Compute the edge capacity for the connected satellites
        capacity_on_graph = self.compute_capacity(
            np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
        )
        prices = self.price_graph.get_prices(connected_sat, sum_both_direction=True)
        
        rates = self.get_rates_dual(costs)
        
        ret_rate_mtch = - np.sum(prices * capacity_on_graph)
        ret_rate_cost = np.sum(rates * (-1 + costs))
        ret = ret_rate_cost + ret_rate_mtch
        if with_prices:
            return ret, np.concatenate((self.possible_sat_pair_expanded,self.price_graph.get_prices(self.possible_sat_pair_expanded,sum_both_direction=True).reshape(-1,1)), axis=1)
        else:
            if with_objective:
                return ret, ret_rate_cost, ret_rate_mtch
            else:
                return ret

    def get_prim_objective(self, with_rates=False):
        self._print("Computing prim objective")
        tic_rounding_matching = self._get_tic()
        connected_sat, connected_lct = self.get_dual_matching()
        tim = self._get_tim(tic_rounding_matching)
        self._add_np_log("prim_rounding_matching_time", self.N_STEP, tim)
        
        tic_rounding_routing = self._get_tic()
        edge_in_both_direction = np.concatenate((connected_sat, connected_sat[:, ::-1]), axis=0)
        prices = self.price_graph.get_prices(edge_in_both_direction)
        indptr, indices, data = build_csr(self.n_sat, edge_in_both_direction, prices, sym_half=False)
        costs, lengths, paths_all = multi_dijkstra_with_paths(self.n_sat, indptr, indices, self.data_source, self.data_target, data)
        srouting = construct_edges_matrix_all_in_one(
            self.data_source, self.data_target, costs, lengths, paths_all
        )
        tim = self._get_tim(tic_rounding_routing)
        self._add_np_log("prim_rounding_routing_time", self.N_STEP, tim)
        
        tic_rounding_rates = self._get_tic()
        rates = self.get_rates_prim(srouting, connected_lct, mode=self.objective_mode)
        self._print(f"Real rate: MAX: {np.max(rates)}, MIN: {np.min(rates)}")
        self._print(f"Appr rate: MAX: {np.max(self.s_t_traffic_rates)}, MIN: {np.min(self.s_t_traffic_rates)}")
        tim = self._get_tim(tic_rounding_rates)
        self._add_np_log("prim_rounding_rates_time", self.N_STEP, tim)
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

        indptr, indices, data = build_csr(self.n_sat, connected_sat, 1/capacity, sym_half=True)
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

    def get_prim_objective_heuristic(self, with_rates=False, matching_method= "mwm", routing_method="ospf",seed=0):
        if matching_method == "grid":
            print("Using grid matching method")
            edge_weights_pr = self.possible_lct_pair_expanded_view_cos[:,0] + self.possible_lct_pair_expanded_view_cos[:,1]
            edge_weights_pr = edge_weights_pr.reshape(-1)
        elif matching_method == "rand":
            print("Using random matching method")
            rng = np.random.default_rng(seed)
            edge_weights_pr = rng.random(self.possible_lct_pair_expanded.shape[0])
        else:
            print("Using other matching method")
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

        if routing_method == "ospf":
            print("Using OSPF routing method")
            indptr, indices, data = build_csr(self.n_sat, connected_sat, 1/capacity, sym_half=True)
        else:
            print("Using other routing method")
            indptr, indices, data = build_csr(self.n_sat, connected_sat, np.ones_like(capacity), sym_half=True)
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
        
    def get_rates_prim(self, srouting, matching, mode="maxsum"):
        # Compute the edge capacity for the connected satellites
        connected_sat = matching // self.N_LCT_PER_SAT
        capacity = self.compute_capacity(
            np.linalg.norm(self.positions[connected_sat[:, 0]] - self.positions[connected_sat[:, 1]], axis=1)
        )
        indptr, indices, data = build_csr(self.n_sat, connected_sat, capacity, merging_method="sum", sym_half=True)
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
        tic = self._get_tic()
        prob.solve(solver=cp.HIGHS, highs_options=dict(solver="simplex"))        
        tim = self._get_tim(tic)
        self._printalltime(f"Prim LP solve took {tim:.2f} us")
        
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
        tic = self._get_tic()
        prob.solve(solver=cp.HIGHS, highs_options=dict(solver="simplex"))        
        tim = self._get_tim(tic)
        self._printalltime(f"Dual LP solve took {tim:.2f} us")
        opt = r.value
        return opt


    def update_step_rates_prices(self):
        pass

if __name__ == '__main__':
    print(np.exp(-np.inf))