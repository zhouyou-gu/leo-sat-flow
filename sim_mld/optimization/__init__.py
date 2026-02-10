"""
Network optimization strategies for satellite link assignment and routing.

This package provides pluggable algorithms for:
- Link matching (assigning LCTs to form ISLs)
- Path routing (finding routes through the network)
- Rate allocation (optimizing traffic flows)

Modules
-------
matching
    Strategies for satellite link matching (greedy, Hungarian, etc.)
routing
    Strategies for network path routing (Dijkstra, OSPF, etc.)
solver
    Network optimization solver coordinating matching and routing
"""

from .matching import (
    MatchingStrategy,
    GreedyMatchingStrategy,
    greedy_max_weight_matching,
)

from .routing import (
    RoutingStrategy,
    DijkstraRoutingStrategy,
)

__all__ = [
    # Matching
    "MatchingStrategy",
    "GreedyMatchingStrategy",
    "greedy_max_weight_matching",
    # Routing
    "RoutingStrategy",
    "DijkstraRoutingStrategy",
]
