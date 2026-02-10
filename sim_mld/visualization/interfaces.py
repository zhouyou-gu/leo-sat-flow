"""
Abstract interfaces for visualization backends.

This module defines the contract that all visualization backends must implement,
enabling swappable rendering engines (Vispy, Matplotlib, custom, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class VisualizationBackend(ABC):
    """
    Abstract base class for visualization backends.
    
    This interface allows the simulation to be rendered using different
    visualization libraries (Vispy, Matplotlib, etc.) without changing
    the core simulation code.
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the visualization backend with configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with backend-specific parameters
        """
        pass
    
    @abstractmethod
    def update_satellite_positions(self, positions: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
        """
        Update satellite marker positions.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of shape (n_sat, 3) with satellite positions
        colors : np.ndarray, optional
            Array of shape (n_sat, 4) with RGBA colors
        """
        pass
    
    @abstractmethod
    def update_satellite_arrows(self, arrow_data: np.ndarray, arrow_colors: np.ndarray) -> None:
        """
        Update LCT direction arrows.
        
        Parameters
        ----------
        arrow_data : np.ndarray
            Array of shape (n_arrows * 2, 3) with arrow start/end positions
        arrow_colors : np.ndarray
            Array of shape (n_arrows * 2, 4) with RGBA colors
        """
        pass
    
    @abstractmethod
    def update_links(self, link_positions: np.ndarray, link_colors: np.ndarray, link_type: str = "isl") -> None:
        """
        Update inter-satellite or ground-satellite link visualization.
        
        Parameters
        ----------
        link_positions : np.ndarray
            Array of shape (n_links * 2, 3) with link endpoint positions
        link_colors : np.ndarray
            Array of shape (n_links * 2, 4) with RGBA colors
        link_type : str
            Type of link: "isl" (inter-satellite) or "gsl" (ground-satellite)
        """
        pass
    
    @abstractmethod
    def update_ground_stations(self, positions: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
        """
        Update ground station marker positions.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of shape (n_gws, 3) with ground station positions
        colors : np.ndarray, optional
            Array of shape (n_gws, 4) with RGBA colors
        """
        pass
    
    @abstractmethod
    def render(self) -> None:
        """
        Render the current frame.
        
        This should update the display with all current visual elements.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all visual elements.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Close the visualization window/context and cleanup resources.
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the visualization is still active.
        
        Returns
        -------
        bool
            True if visualization window is open, False otherwise
        """
        pass


class NullVisualizationBackend(VisualizationBackend):
    """
    Null object pattern implementation for headless simulations.
    
    This backend does nothing, allowing the simulation to run without
    any visualization overhead.
    """
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """No-op initialization."""
        pass
    
    def update_satellite_positions(self, positions: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
        """No-op update."""
        pass
    
    def update_satellite_arrows(self, arrow_data: np.ndarray, arrow_colors: np.ndarray) -> None:
        """No-op update."""
        pass
    
    def update_links(self, link_positions: np.ndarray, link_colors: np.ndarray, link_type: str = "isl") -> None:
        """No-op update."""
        pass
    
    def update_ground_stations(self, positions: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
        """No-op update."""
        pass
    
    def render(self) -> None:
        """No-op render."""
        pass
    
    def clear(self) -> None:
        """No-op clear."""
        pass
    
    def close(self) -> None:
        """No-op close."""
        pass
    
    def is_running(self) -> bool:
        """Always returns True for headless mode."""
        return True
