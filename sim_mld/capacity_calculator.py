"""
Capacity calculation module for optical inter-satellite links.

This module provides functionality for computing link capacities based on
distance and optical channel characteristics.
"""

import numpy as np
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
