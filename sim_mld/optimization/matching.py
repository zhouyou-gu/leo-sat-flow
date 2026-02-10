"""
Matching strategies for satellite link assignment.

This module provides different strategies for matching satellites to form
inter-satellite links, including greedy and maximum weight matching approaches.

The matching problem is to assign laser communication terminals (LCTs) on
satellites to form inter-satellite links (ISLs) such that:
1. Each LCT is connected to at most one other LCT
2. The total link quality/capacity is maximized
3. Physical constraints (field of view, distance) are satisfied

Supported Strategies
--------------------
- Greedy Matching: Fast heuristic that selects edges in order of weight
- (Future) Hungarian Algorithm: Optimal matching in polynomial time
- (Future) Network Flow: General optimization-based matching

Usage Examples
--------------
    from sim_mld.matching_strategy import GreedyMatchingStrategy
    import numpy as np
    
    # Create weighted edges: [source, target, weight]
    edges = np.array([
        [0, 1, 10.5],  # Satellite 0 to 1, weight 10.5
        [1, 2, 8.3],   # Satellite 1 to 2, weight 8.3
        [0, 2, 5.1],   # Satellite 0 to 2, weight 5.1
    ])
    
    # Compute matching
    strategy = GreedyMatchingStrategy()
    matched_pairs = strategy.compute_matching(edges)
    # Result: [[0, 1], [2, ...]] (best matches)

Design Pattern
--------------
This module uses the Strategy pattern to enable:
- Runtime selection of matching algorithms
- Easy addition of new algorithms
- Testability and benchmarking of different approaches
"""

import numpy as np
from numba import njit


@njit(cache=True)
def greedy_max_weight_matching(E: np.ndarray) -> list:
    """
    Compute a greedy heuristic maximum weight matching for a NumPy array of edges.
    
    This algorithm sorts edges by weight in descending order and greedily selects
    edges where both endpoints are unmatched.
    
    Parameters
    ----------
    E : np.ndarray
        Array with shape (num_edges, 3) where each row is [u, v, weight].
    
    Returns
    -------
    list
        List of tuples (u, v) representing the selected edges.
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


class MatchingStrategy:
    """
    Base class for matching strategies.
    
    This class defines the interface for different matching algorithms
    that assign satellite pairs to form inter-satellite links.
    """
    
    def compute_matching(self, weighted_edges):
        """
        Compute a matching given weighted edges.
        
        Parameters
        ----------
        weighted_edges : np.ndarray
            Array with shape (num_edges, 3) where each row is [u, v, weight]
            
        Returns
        -------
        np.ndarray
            Array of matched satellite pairs, shape (num_matches, 2)
        """
        raise NotImplementedError("Subclasses must implement compute_matching")


class GreedyMatchingStrategy(MatchingStrategy):
    """
    Greedy matching strategy that selects edges in order of decreasing weight.
    """
    
    def compute_matching(self, weighted_edges):
        """
        Compute a greedy maximum weight matching.
        
        Parameters
        ----------
        weighted_edges : np.ndarray
            Array with shape (num_edges, 3) where each row is [u, v, weight]
            
        Returns
        -------
        np.ndarray
            Array of matched satellite pairs, shape (num_matches, 2)
        """
        matching_list = greedy_max_weight_matching(weighted_edges)
        if len(matching_list) == 0:
            return np.array([]).reshape(0, 2)
        return np.array(matching_list)
