"""
Abstract interfaces for network link models.

This module defines interfaces for pluggable link capacity and physics models,
enabling different channel models (optical, RF, custom) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class LinkCapacityModel(ABC):
    """
    Abstract base class for link capacity computation models.
    
    This interface allows different channel models (optical Gaussian beam,
    RF link budget, custom ML-based, etc.) to be plugged into the simulation.
    """
    
    @abstractmethod
    def compute_capacity(self, distance: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute link capacity based on distance and other parameters.
        
        Parameters
        ----------
        distance : np.ndarray
            Distance(s) in meters between link endpoints
        **kwargs : dict
            Additional model-specific parameters (angles, weather, etc.)
            
        Returns
        -------
        np.ndarray
            Link capacity in Gbps for each distance
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters for logging/debugging.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of model parameters (wavelength, power, etc.)
        """
        pass
    
    def compute_capacity_from_positions(
        self, 
        positions: np.ndarray, 
        sat_pair_indices: np.ndarray,
        earth_radius: float = 6371e3
    ) -> np.ndarray:
        """
        Compute link capacities from satellite positions and pair indices.
        
        This is a convenience method that handles distance calculation.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of satellite positions, shape (n_sat, 3)
        sat_pair_indices : np.ndarray
            Array of satellite pair indices, shape (n_pairs, 2)
        earth_radius : float
            Earth radius in meters for distance scaling
            
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
        satellite_distances = satellite_distances * earth_radius
        return self.compute_capacity(satellite_distances)


class LinkPhysicsModel(ABC):
    """
    Abstract base class for link physics models.
    
    This interface provides detailed physical layer modeling beyond just capacity,
    including signal strength, SNR, BER, etc.
    """
    
    @abstractmethod
    def compute_received_power(self, distance: np.ndarray, tx_power: float, **kwargs) -> np.ndarray:
        """
        Compute received power at the receiver.
        
        Parameters
        ----------
        distance : np.ndarray
            Distance(s) in meters
        tx_power : float
            Transmit power in Watts
        **kwargs : dict
            Additional parameters (wavelength, beam divergence, etc.)
            
        Returns
        -------
        np.ndarray
            Received power in Watts
        """
        pass
    
    @abstractmethod
    def compute_snr(self, distance: np.ndarray, tx_power: float, **kwargs) -> np.ndarray:
        """
        Compute signal-to-noise ratio.
        
        Parameters
        ----------
        distance : np.ndarray
            Distance(s) in meters
        tx_power : float
            Transmit power in Watts
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            SNR in dB
        """
        pass


class OpticalGaussianLinkModel(LinkCapacityModel):
    """
    Optical link capacity model using Gaussian beam physics.
    
    This model accounts for beam divergence, pointing jitter, and atmospheric effects.
    It uses Shannon capacity with optical signal characteristics.
    """
    
    def __init__(
        self,
        wavelength: float = 1.55e-6,
        angular_spreading: float = 100e-6,
        peak_power: float = 20.0,
        bandwidth: float = 1e9,
        responsivity: float = 0.5,
        aperture_area: float = 1e-2,
        noise_current: float = 3e-7,
        jitter: float = 10e-6,
        epsilon: float = 1e-3
    ):
        """
        Initialize optical Gaussian link model.
        
        Parameters
        ----------
        wavelength : float
            Laser wavelength in meters (default: 1.55 μm telecom)
        angular_spreading : float
            Beam angular spreading in radians (default: 100 μrad)
        peak_power : float
            Peak transmit power in Watts (default: 20 W)
        bandwidth : float
            Signal bandwidth in Hz (default: 1 GHz)
        responsivity : float
            Photodetector responsivity in A/W (default: 0.5)
        aperture_area : float
            Receiver aperture area in m² (default: 1 cm²)
        noise_current : float
            Receiver noise current in A rms (default: 0.3 μA)
        jitter : float
            Pointing jitter in radians (default: 10 μrad)
        epsilon : float
            Reliability margin (default: 0.1%)
        """
        self.wavelength = wavelength
        self.angular_spreading = angular_spreading
        self.peak_power = peak_power
        self.bandwidth = bandwidth
        self.responsivity = responsivity
        self.aperture_area = aperture_area
        self.noise_current = noise_current
        self.jitter = jitter
        self.epsilon = epsilon
        
        # Compute derived parameters
        from sim_mld.network.channel_model import w0_from_angular_spreading, capacity_relaxed
        self.beam_waist = w0_from_angular_spreading(angular_spreading, wavelength)
        self._capacity_relaxed = capacity_relaxed
    
    def compute_capacity(self, distance: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute optical link capacity using Gaussian beam model.
        
        Parameters
        ----------
        distance : np.ndarray
            Distance in meters
        **kwargs : dict
            Ignored for this model
            
        Returns
        -------
        np.ndarray
            Link capacity in Gbps
        """
        return self._capacity_relaxed(
            self.bandwidth,
            self.responsivity,
            self.aperture_area,
            self.noise_current,
            self.peak_power,
            self.beam_waist,
            self.wavelength,
            distance,
            self.jitter,
            self.epsilon
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "model_type": "optical_gaussian",
            "wavelength_um": self.wavelength * 1e6,
            "angular_spreading_urad": self.angular_spreading * 1e6,
            "peak_power_W": self.peak_power,
            "bandwidth_GHz": self.bandwidth / 1e9,
            "beam_waist_m": self.beam_waist
        }
