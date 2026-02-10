"""
Capacity calculation module for optical inter-satellite links.

This module provides functionality for computing link capacities based on
distance and optical channel characteristics.

The capacity calculations are based on Shannon's theorem applied to optical
communication channels with Gaussian beam characteristics, accounting for
factors such as:
- Beam divergence and spreading
- Atmospheric/pointing jitter
- Receiver aperture size
- Noise characteristics

Usage Examples
--------------
    from sim_mld.capacity_calculator import CapacityCalculator
    
    # Calculate capacity for a specific distance
    distance_m = 1500e3  # 1500 km
    capacity_gbps = CapacityCalculator.compute_capacity(distance_m)
    
    # Calculate capacities for multiple satellite pairs
    import numpy as np
    positions = np.random.randn(100, 3)  # 100 satellites
    sat_pairs = np.array([[0, 1], [2, 3], [4, 5]])  # 3 pairs
    capacities = CapacityCalculator.compute_capacity_from_positions(
        positions, sat_pairs
    )

Notes
-----
The capacity model includes:
- Wavelength-dependent beam characteristics (1.55 μm telecom band)
- Pointing jitter effects (10 μrad typical)
- Relaxed capacity with epsilon tolerance (0.1% reliability margin)
"""

import numpy as np

try:
    from sim_mld.network.channel_model import (
        w0_from_angular_spreading,
        capacity_relaxed,
    )
    from sim_mld.core.constants import (
        OPTICAL_RESPONSIVITY,
        EARTH_RADIUS_KM,
    )
except ImportError:
    # Fallback for backward compatibility
    from sim_mld.lisl_channel_model import (
        w0_from_angular_spreading,
        capacity_relaxed,
    )
    from sim_mld.constants import (
        OPTICAL_RESPONSIVITY,
        EARTH_RADIUS_KM,
    )


class CapacityCalculator:
    """
    Calculator for optical inter-satellite link capacities.
    
    This class encapsulates the physics of optical communication channels
    and provides methods to compute link capacities based on distance and
    channel characteristics.
    """
    
    # Physical parameters for optical communication
    WAVELENGTH = 1.55e-6  # Wavelength in meters (1.55 microns typical in telecom)
    ANGULAR_SPREADING = 100e-6  # Angular spreading in radians
    BEAM_WAIST = w0_from_angular_spreading(ANGULAR_SPREADING, WAVELENGTH)  # Beam waist in meters
    PEAK_POWER_W = 20  # Peak power in Watts
    BANDWIDTH = 1e9  # 1 GHz bandwidth
    RESPONSIVITY = OPTICAL_RESPONSIVITY  # A/W responsivity
    APERTURE_AREA = 1e-2  # Aperture area in m^2
    NOISE_CURRENT = 3e-7  # Noise current in A rms
    JITTER = 10e-6  # Pointing jitter in radians
    EPSILON = 1e-3  # Epsilon for relaxed capacity calculations
    
    # Note: EARTH_RADIUS is kept here in meters to match the distance units
    # expected by capacity calculations. The constant module stores it in km
    # for general use, but this class needs it in meters for internal consistency.
    EARTH_RADIUS = EARTH_RADIUS_KM * 1e3  # Earth radius in meters
    MIN_CAPACITY = 1  # Minimum capacity threshold in Gbps
    
    @classmethod
    def compute_capacity(cls, distance):
        """
        Compute the link capacity based on distance.
        
        Parameters
        ----------
        distance : float or np.ndarray
            Distance in meters between satellites
            
        Returns
        -------
        float or np.ndarray
            Link capacity in Gbps
        """
        C = capacity_relaxed(
            cls.BANDWIDTH,
            cls.RESPONSIVITY,
            cls.APERTURE_AREA,
            cls.NOISE_CURRENT,
            cls.PEAK_POWER_W,
            cls.BEAM_WAIST,
            cls.WAVELENGTH,
            distance,
            cls.JITTER,
            cls.EPSILON
        )
        return C
    
    @classmethod
    def compute_capacity_from_positions(cls, positions, sat_pair_indices):
        """
        Compute link capacities from satellite positions and pair indices.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of satellite positions, shape (n_sat, 3)
        sat_pair_indices : np.ndarray
            Array of satellite pair indices, shape (n_pairs, 2)
            
        Returns
        -------
        np.ndarray
            Link capacities in Gbps for each pair
        """
        satellite_distances = np.linalg.norm(
            positions[sat_pair_indices[:, 0]] - positions[sat_pair_indices[:, 1]], 
            axis=1
        )
        # Convert to meters
        satellite_distances = satellite_distances * cls.EARTH_RADIUS
        return cls.compute_capacity(satellite_distances)
