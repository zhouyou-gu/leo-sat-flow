"""
Configuration schema and validation for sim_mld.

This module provides dataclasses for configuration with validation and type safety.
It enables easy configuration management and parameter profiles.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


class CapacityModelType(str, Enum):
    """Supported link capacity models."""
    OPTICAL_GAUSSIAN = "optical_gaussian"
    RF_LINK_BUDGET = "rf_link_budget"
    CUSTOM = "custom"


class MatchingStrategyType(str, Enum):
    """Supported matching strategies."""
    GREEDY = "greedy"
    HUNGARIAN = "hungarian"
    NETWORK_FLOW = "network_flow"


class RoutingStrategyType(str, Enum):
    """Supported routing strategies."""
    DIJKSTRA = "dijkstra"
    OSPF = "ospf"
    LOAD_BALANCED = "load_balanced"


class ConstellationType(str, Enum):
    """Pre-defined constellation types."""
    STARLINK = "starlink"
    KUIPER = "kuiper"
    ONEWEB = "oneweb"
    CUSTOM = "custom"


@dataclass
class NetworkConfig:
    """
    Configuration for network topology and link characteristics.
    
    Attributes
    ----------
    n_lct_per_sat : int
        Number of Laser Communication Terminals per satellite (default: 4)
    max_lisl_distance_km : float
        Maximum inter-satellite link distance in kilometers (default: 3000)
    for_theta_half_deg : float
        Field of view half-angle in degrees (default: 60)
    link_capacity_model : CapacityModelType
        Type of link capacity model to use
    """
    n_lct_per_sat: int = 4
    max_lisl_distance_km: float = 3000.0
    for_theta_half_deg: float = 60.0
    link_capacity_model: CapacityModelType = CapacityModelType.OPTICAL_GAUSSIAN
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_lct_per_sat < 1:
            raise ValueError(f"n_lct_per_sat must be >= 1, got {self.n_lct_per_sat}")
        if self.max_lisl_distance_km <= 0:
            raise ValueError(f"max_lisl_distance_km must be > 0, got {self.max_lisl_distance_km}")
        if not (0 < self.for_theta_half_deg < 180):
            raise ValueError(f"for_theta_half_deg must be in (0, 180), got {self.for_theta_half_deg}")


@dataclass
class SolverConfig:
    """
    Configuration for network optimization solver.
    
    Attributes
    ----------
    matching_strategy : MatchingStrategyType
        Strategy for link matching (default: greedy)
    routing_strategy : RoutingStrategyType
        Strategy for path routing (default: dijkstra)
    objective_mode : Literal["maxsum", "maxmin"]
        Optimization objective (default: "maxsum")
    alpha : float
        Step size parameter for iterative algorithms (default: 0.1)
    beta : float
        Optical efficiency factor (default: 0.5)
    """
    matching_strategy: MatchingStrategyType = MatchingStrategyType.GREEDY
    routing_strategy: RoutingStrategyType = RoutingStrategyType.DIJKSTRA
    objective_mode: Literal["maxsum", "maxmin"] = "maxsum"
    alpha: float = 0.1
    beta: float = 0.5
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0 < self.alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not (0 < self.beta <= 1):
            raise ValueError(f"beta must be in (0, 1], got {self.beta}")


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization rendering.
    
    Attributes
    ----------
    backend : Literal["vispy", "matplotlib", "none"]
        Visualization backend to use
    plot_satellite_lcts : bool
        Whether to show LCT direction arrows
    plot_potential_lisl : bool
        Whether to show potential ISL edges
    plot_gws_point_size : float
        Ground station marker size
    plot_sat_point_size : float
        Satellite marker size
    """
    backend: Literal["vispy", "matplotlib", "none"] = "vispy"
    plot_satellite_lcts: bool = True
    plot_potential_lisl: bool = False
    plot_gws_point_size: float = 10.0
    plot_sat_point_size: float = 10.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.plot_gws_point_size <= 0:
            raise ValueError(f"plot_gws_point_size must be > 0, got {self.plot_gws_point_size}")
        if self.plot_sat_point_size <= 0:
            raise ValueError(f"plot_sat_point_size must be > 0, got {self.plot_sat_point_size}")


@dataclass
class SimulationConfig:
    """
    Top-level configuration for simulation.
    
    Attributes
    ----------
    constellation_type : ConstellationType
        Type of constellation to simulate
    time_scale : float
        Simulation time scale factor (default: 15.0)
    network : NetworkConfig
        Network topology configuration
    solver : SolverConfig
        Optimization solver configuration
    visualization : Optional[VisualizationConfig]
        Visualization configuration (None for headless)
    """
    constellation_type: ConstellationType = ConstellationType.STARLINK
    time_scale: float = 15.0
    network: NetworkConfig = field(default_factory=NetworkConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    visualization: Optional[VisualizationConfig] = field(default_factory=VisualizationConfig)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.time_scale <= 0:
            raise ValueError(f"time_scale must be > 0, got {self.time_scale}")
    
    @classmethod
    def from_profile(cls, profile: str) -> "SimulationConfig":
        """
        Create configuration from a pre-defined profile.
        
        Parameters
        ----------
        profile : str
            Profile name ("starlink", "kuiper", "oneweb", "minimal")
            
        Returns
        -------
        SimulationConfig
            Configuration instance for the specified profile
        """
        profiles = {
            "starlink": cls(
                constellation_type=ConstellationType.STARLINK,
                network=NetworkConfig(n_lct_per_sat=4, max_lisl_distance_km=5000),
                time_scale=15.0
            ),
            "kuiper": cls(
                constellation_type=ConstellationType.KUIPER,
                network=NetworkConfig(n_lct_per_sat=4, max_lisl_distance_km=4000),
                time_scale=15.0
            ),
            "oneweb": cls(
                constellation_type=ConstellationType.ONEWEB,
                network=NetworkConfig(n_lct_per_sat=2, max_lisl_distance_km=3000),
                time_scale=10.0
            ),
            "minimal": cls(
                constellation_type=ConstellationType.CUSTOM,
                network=NetworkConfig(n_lct_per_sat=2, max_lisl_distance_km=2000),
                visualization=None  # Headless
            )
        }
        
        if profile not in profiles:
            raise ValueError(f"Unknown profile '{profile}'. Available: {list(profiles.keys())}")
        
        return profiles[profile]
