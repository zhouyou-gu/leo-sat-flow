"""
Visualization backends for satellite constellation rendering.

This package provides pluggable visualization backends enabling different
rendering engines (Vispy, Matplotlib, etc.) without changing core simulation code.

Modules
-------
interfaces
    Abstract base classes for visualization backends
"""

from .interfaces import (
    VisualizationBackend,
    NullVisualizationBackend,
)

__all__ = [
    "VisualizationBackend",
    "NullVisualizationBackend",
]
