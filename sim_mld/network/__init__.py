"""
Network layer abstractions for satellite constellation simulations.

This package handles network topology, link capacity models, and graph operations.

Modules
-------
interfaces
    Abstract base classes for link capacity and physics models
channel_model
    Optical and RF channel models for link capacity
capacity
    Capacity calculator implementations
graph
    Graph operations (CSR, Dijkstra, shortest paths)
"""

from .interfaces import (
    LinkCapacityModel,
    LinkPhysicsModel,
    OpticalGaussianLinkModel,
)

__all__ = [
    "LinkCapacityModel",
    "LinkPhysicsModel",
    "OpticalGaussianLinkModel",
]
