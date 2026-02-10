"""
Configuration schemas and validation for sim_mld.

This package provides type-safe configuration with dataclasses,
enabling easy parameter management and validation.

Modules
-------
schema
    Configuration dataclasses (NetworkConfig, SolverConfig, etc.)
"""

from .schema import (
    NetworkConfig,
    SolverConfig,
    VisualizationConfig,
    SimulationConfig,
    CapacityModelType,
    MatchingStrategyType,
    RoutingStrategyType,
    ConstellationType,
)

__all__ = [
    "NetworkConfig",
    "SolverConfig",
    "VisualizationConfig",
    "SimulationConfig",
    "CapacityModelType",
    "MatchingStrategyType",
    "RoutingStrategyType",
    "ConstellationType",
]
