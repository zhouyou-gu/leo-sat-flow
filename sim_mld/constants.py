"""
Constants and configuration values for the LEO satellite flow simulation.

This module centralizes magic numbers and configuration parameters that were
previously scattered throughout the codebase to improve maintainability.
"""

# Mathematical and computational constants
INFINITY_THRESHOLD = 1e12  # Large value representing infinity in path costs
LARGE_NUMBER = 1e10  # Large value for approximate infinity in view times
MIDPOINT_FACTOR = 0.5  # Factor for computing midpoint between two points
POSITION_LIFT_FACTOR = 1.0001  # Small factor to lift visualization positions above surface

# Physical constants
EARTH_RADIUS_KM = 6371.0  # Earth's radius in kilometers

# Optical communication parameters
OPTICAL_RESPONSIVITY = 0.5  # A/W responsivity for optical receivers
OPTICAL_EFFICIENCY_BETA = 0.5  # Optical efficiency factor

# Satellite configuration
DEFAULT_LCT_COUNT = 4  # Default number of Laser Communication Terminals per satellite
DEFAULT_FOR_THETA_HALF = 60.0  # Default field of view half-angle in degrees
DEFAULT_LISL_MAX_DISTANCE = 3000.0  # Default maximum inter-satellite link distance in km

# Visualization parameters
DEFAULT_ARROW_SCALE_FACTOR = 0.005  # Scale factor for direction arrows in visualization
DEFAULT_ALPHA_TRANSPARENCY = 0.5  # Default transparency level for edges

# Simulation timing
DEFAULT_TIME_SCALE = 15.0  # Default simulation time scale factor

# Network optimization
DEFAULT_WEIGHT_THRESHOLD = 0.5  # Threshold for determining edge weights in routing
