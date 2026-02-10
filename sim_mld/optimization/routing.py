"""
Routing strategies for satellite network traffic.

This module provides different routing strategies for computing paths
through the satellite network, including shortest path algorithms.

The routing problem is to find paths for data flows from source satellites
to destination satellites through the inter-satellite link network. The goal
is to minimize path cost (distance, latency, congestion) while ensuring
connectivity.

Supported Strategies
--------------------
- Dijkstra Routing: Single-source shortest path (SSSP) algorithm
- (Future) OSPF: Open Shortest Path First with link state
- (Future) Load-Balanced: Distribute traffic across multiple paths
- (Future) QoS-Aware: Route with quality of service constraints

Usage Examples
--------------
    from sim_mld.routing_strategy import DijkstraRoutingStrategy
    import numpy as np
    
    # Network graph definition
    n_satellites = 10
    edges = np.array([[0, 1], [1, 2], [2, 3], ...])  # Connectivity
    costs = np.array([100.5, 150.2, 200.1, ...])     # Edge costs (distance/latency)
    
    # Traffic demands
    sources = np.array([0, 2, 4])   # Source satellites
    targets = np.array([5, 7, 9])   # Destination satellites
    
    # Compute routes
    strategy = DijkstraRoutingStrategy()
    costs, lengths, paths = strategy.compute_routes(
        n_satellites, edges, costs, sources, targets
    )
    
    # Filter unreachable pairs
    filtered = strategy.filter_unreachable_pairs(
        costs, lengths, sources, targets
    )

Design Pattern
--------------
This module uses the Strategy pattern to enable:
- Runtime selection of routing algorithms
- Easy addition of new algorithms (OSPF, load balancing)
- Benchmarking and comparison of different approaches
- Testability in isolation

Performance Notes
-----------------
The Dijkstra implementation uses:
- CSR (Compressed Sparse Row) graph representation for efficiency
- Numba JIT compilation for parallel processing
- Multi-query optimization to route many flows simultaneously
"""

import numpy as np
from sim_mld.dijkstra import build_csr, multi_dijkstra_with_paths
from sim_mld.constants import INFINITY_THRESHOLD


class RoutingStrategy:
    """
    Base class for routing strategies.
    
    This class defines the interface for different routing algorithms
    that compute paths through the satellite network.
    """
    
    def compute_routes(self, n_sat, sat_pair_edges, edge_costs, sources, targets):
        """
        Compute routes for given source-target pairs.
        
        Parameters
        ----------
        n_sat : int
            Number of satellites in the network
        sat_pair_edges : np.ndarray
            Array of satellite pair indices, shape (n_pairs, 2)
        edge_costs : np.ndarray
            Array of edge costs/weights, shape (n_pairs,)
        sources : np.ndarray
            Array of source satellite indices
        targets : np.ndarray
            Array of target satellite indices
            
        Returns
        -------
        tuple
            (costs, lengths, paths_all) where:
            - costs: path costs for each source-target pair
            - lengths: path lengths for each source-target pair
            - paths_all: actual paths for each source-target pair
        """
        raise NotImplementedError("Subclasses must implement compute_routes")


class DijkstraRoutingStrategy(RoutingStrategy):
    """
    Routing strategy using Dijkstra's shortest path algorithm.
    
    This strategy computes shortest paths based on edge costs using
    the Dijkstra algorithm with efficient CSR graph representation.
    """
    
    def compute_routes(self, n_sat, sat_pair_edges, edge_costs, sources, targets):
        """
        Compute shortest paths using Dijkstra's algorithm.
        
        Parameters
        ----------
        n_sat : int
            Number of satellites in the network
        sat_pair_edges : np.ndarray
            Array of satellite pair indices, shape (n_pairs, 2)
        edge_costs : np.ndarray
            Array of edge costs/weights, shape (n_pairs,)
        sources : np.ndarray
            Array of source satellite indices
        targets : np.ndarray
            Array of target satellite indices
            
        Returns
        -------
        tuple
            (costs, lengths, paths_all) where:
            - costs: path costs for each source-target pair
            - lengths: path lengths for each source-target pair
            - paths_all: actual paths for each source-target pair
        """
        # Build CSR representation for efficient graph operations
        indptr, indices, data = build_csr(n_sat, sat_pair_edges, edge_costs, sym_half=False)
        
        # Compute multiple shortest paths in parallel
        costs, lengths, paths_all = multi_dijkstra_with_paths(
            n_sat, indptr, indices, sources, targets, data
        )
        
        return costs, lengths, paths_all
    
    def filter_unreachable_pairs(self, costs, lengths, sources, targets, traffic_rates=None):
        """
        Filter out unreachable source-target pairs.
        
        Parameters
        ----------
        costs : np.ndarray
            Path costs from routing
        lengths : np.ndarray
            Path lengths from routing
        sources : np.ndarray
            Source satellite indices
        targets : np.ndarray
            Target satellite indices
        traffic_rates : np.ndarray, optional
            Traffic rates for each pair (will be filtered as well)
            
        Returns
        -------
        tuple
            Filtered (sources, targets, costs, traffic_rates) with only reachable pairs
        """
        # Filter pairs that are reachable (cost less than infinity threshold)
        # Note: length > 0 is not sufficient as it could include unreachable paths
        valid_mask = (lengths > 0) & (costs < INFINITY_THRESHOLD)
        filtered_sources = sources[valid_mask]
        filtered_targets = targets[valid_mask]
        filtered_costs = costs[valid_mask]
        
        if traffic_rates is not None:
            filtered_traffic_rates = traffic_rates[valid_mask]
            return filtered_sources, filtered_targets, filtered_costs, filtered_traffic_rates
        
        return filtered_sources, filtered_targets, filtered_costs
