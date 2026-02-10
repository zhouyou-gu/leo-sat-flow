"""
Routing strategies for satellite network traffic.

This module provides different routing strategies for computing paths
through the satellite network, including shortest path algorithms.
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
        # Filter pairs that have valid paths (length > 0)
        valid_mask = lengths > 0
        filtered_sources = sources[valid_mask]
        filtered_targets = targets[valid_mask]
        filtered_costs = costs[valid_mask]
        
        if traffic_rates is not None:
            filtered_traffic_rates = traffic_rates[valid_mask]
            return filtered_sources, filtered_targets, filtered_costs, filtered_traffic_rates
        
        return filtered_sources, filtered_targets, filtered_costs
