"""
Core mathematical and geometric functions for satellite simulations.

This package provides pure mathematical functions with minimal dependencies,
forming the foundation for all spatial computations.

Modules
-------
coordinates
    Coordinate system transformations (spherical ↔ Cartesian, rotations)
constants
    Physical and mathematical constants used throughout the simulation
"""

from .coordinates import (
    lat_lon_to_xyz,
    xyz_to_lat_lon,
    rotate_deg_in_vector,
    rotate_deg_in_vector_element_wise,
    rotation_matmul,
)

from .constants import (
    INFINITY_THRESHOLD,
    LARGE_NUMBER,
    MIDPOINT_FACTOR,
    POSITION_LIFT_FACTOR,
    EARTH_RADIUS_KM,
    OPTICAL_RESPONSIVITY,
    OPTICAL_EFFICIENCY_BETA,
    DEFAULT_LCT_COUNT,
    DEFAULT_FOR_THETA_HALF,
    DEFAULT_LISL_MAX_DISTANCE,
    DEFAULT_TIME_SCALE,
    DEFAULT_ARROW_SCALE_FACTOR,
    DEFAULT_ALPHA_TRANSPARENCY,
    DEFAULT_WEIGHT_THRESHOLD,
)

__all__ = [
    # Coordinate functions
    "lat_lon_to_xyz",
    "xyz_to_lat_lon",
    "rotate_deg_in_vector",
    "rotate_deg_in_vector_element_wise",
    "rotation_matmul",
    # Constants
    "INFINITY_THRESHOLD",
    "LARGE_NUMBER",
    "MIDPOINT_FACTOR",
    "POSITION_LIFT_FACTOR",
    "EARTH_RADIUS_KM",
    "OPTICAL_RESPONSIVITY",
    "OPTICAL_EFFICIENCY_BETA",
    "DEFAULT_LCT_COUNT",
    "DEFAULT_FOR_THETA_HALF",
    "DEFAULT_LISL_MAX_DISTANCE",
    "DEFAULT_TIME_SCALE",
    "DEFAULT_ARROW_SCALE_FACTOR",
    "DEFAULT_ALPHA_TRANSPARENCY",
    "DEFAULT_WEIGHT_THRESHOLD",
]
