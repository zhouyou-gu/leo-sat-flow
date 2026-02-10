"""
Core coordinate transformation functions for satellite constellation geometry.

This module provides pure mathematical functions for coordinate system conversions
and vector rotations. It has zero dependencies beyond NumPy and Numba, making it
the foundation for all spatial computations.

Functions
---------
- lat_lon_to_xyz: Convert spherical to Cartesian coordinates
- xyz_to_lat_lon: Convert Cartesian to spherical coordinates
- rotate_deg_in_vector: Rotate vectors around an axis (batch)
- rotate_deg_in_vector_element_wise: Rotate vectors with per-element angles
- rotation_matmul: Matrix multiplication with 3x3 rotation matrix
"""

from numba import njit, prange
import numpy as np
import math


@njit(cache=True)
def _apply_rodrigues_formula(x, y, z, cos_angle, sin_angle, ux, uy, uz):
    """
    Apply Rodrigues' rotation formula to rotate a single vector.
    
    This is a helper function that applies the core rotation mathematics.
    Rodrigues' formula rotates a vector v by angle θ around axis u:
    v_rot = v*cos(θ) + (u×v)*sin(θ) + u*(u·v)*(1-cos(θ))
    
    Parameters
    ----------
    x, y, z : float
        Components of the vector to rotate
    cos_angle : float
        Cosine of the rotation angle
    sin_angle : float
        Sine of the rotation angle
    ux, uy, uz : float
        Components of the normalized rotation axis
        
    Returns
    -------
    tuple of float
        The rotated vector components (rx, ry, rz)
    """
    rx = (x * (cos_angle + ux * ux * (1 - cos_angle)) +
          y * (ux * uy * (1 - cos_angle) - uz * sin_angle) + 
          z * (ux * uz * (1 - cos_angle) + uy * sin_angle))
    ry = (x * (uy * ux * (1 - cos_angle) + uz * sin_angle) +
          y * (cos_angle + uy * uy * (1 - cos_angle)) +
          z * (uy * uz * (1 - cos_angle) - ux * sin_angle))
    rz = (x * (uz * ux * (1 - cos_angle) - uy * sin_angle) +
          y * (uz * uy * (1 - cos_angle) + ux * sin_angle) +
          z * (cos_angle + uz * uz * (1 - cos_angle)))
    return rx, ry, rz


@njit(parallel=True, cache=True)
def lat_lon_to_xyz(lat_lon, r=1.):
    """
    Convert latitude and longitude arrays (in radians) to 3D Cartesian coordinates on a sphere.

    Uses the standard spherical-to-Cartesian transformation:
    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)

    Parameters
    ----------
    lat_lon : np.ndarray
        Array of shape (n, 2) where each row is [latitude, longitude] in radians.
    r : float, optional
        Radius of the sphere (default is 1.).

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) with Cartesian coordinates [x, y, z].
                
    Examples
    --------
    >>> import numpy as np
    >>> lat_lon = np.array([[0, 0], [np.pi/2, 0]])  # Equator, North pole
    >>> xyz = lat_lon_to_xyz(lat_lon)
    >>> print(xyz)
    [[1. 0. 0.]
     [0. 0. 1.]]
    """
    n = lat_lon.shape[0]
    xyz = np.empty((n, 3), dtype=lat_lon.dtype)
    
    for i in prange(n):
        lat = lat_lon[i, 0]
        lon = lat_lon[i, 1]
        xyz[i, 0] = r * math.cos(lat) * math.cos(lon)  # x
        xyz[i, 1] = r * math.cos(lat) * math.sin(lon)  # y
        xyz[i, 2] = r * math.sin(lat)  # z

    return xyz


@njit(parallel=True, cache=True)
def xyz_to_lat_lon(xyz: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (lat, lon).

    Uses inverse transformation:
    r = sqrt(x² + y² + z²)
    lat = arcsin(z/r)
    lon = arctan2(y, x)

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (n, 3) with Cartesian coordinates [x, y, z].
        
    Returns
    -------
    np.ndarray
        Array of shape (n, 2) with spherical coordinates [latitude, longitude] in radians.
        
    Examples
    --------
    >>> import numpy as np
    >>> xyz = np.array([[1, 0, 0], [0, 0, 1]])  # x-axis, z-axis
    >>> lat_lon = xyz_to_lat_lon(xyz)
    >>> print(lat_lon)
    [[0.         0.        ]
     [1.57079633 0.        ]]  # [0, 0], [π/2, 0]
    """
    n = xyz.shape[0]
    lat_lon = np.empty((n, 2), dtype=xyz.dtype)
    
    for i in prange(n):
        x = xyz[i, 0]
        y = xyz[i, 1]
        z = xyz[i, 2]
        r = math.sqrt(x * x + y * y + z * z)
        if r == 0:
            lat_lon[i, 0] = 0.0
            lat_lon[i, 1] = 0.0
        else:
            x /= r
            y /= r
            z /= r
            # Compute latitude and longitude
            lat_lon[i, 0] = math.asin(z)  # latitude
            lat_lon[i, 1] = math.atan2(y, x)  # longitude
    return lat_lon


@njit(parallel=True, cache=True)
def rotate_deg_in_vector(vectors: np.ndarray, angle_deg: float, angular_vector: np.ndarray = None) -> np.ndarray:
    '''
    Rotate vectors by a given angle around a specified angular vector.
    
    All vectors are rotated by the same angle around the same axis.
    Uses Rodrigues' rotation formula for efficiency.
    
    Parameters
    ----------
    vectors : np.ndarray
        Array of shape (n, 3) containing vectors to be rotated.
    angle_deg : float
        Angle in degrees by which to rotate the vectors.
    angular_vector : np.ndarray, optional
        Angular vector of shape (3,) around which to rotate the vectors.
        Defaults to [0, 0, 1] (z-axis) if not provided.
        
    Returns
    -------
    np.ndarray
        Array of shape (n, 3) containing the rotated vectors.
        
    Examples
    --------
    >>> import numpy as np
    >>> vectors = np.array([[1, 0, 0], [0, 1, 0]])
    >>> # Rotate 90 degrees around z-axis
    >>> rotated = rotate_deg_in_vector(vectors, 90.0)
    >>> print(rotated)  # [0, 1, 0], [-1, 0, 0]
    '''
    # Initialize default angular vector if not provided
    if angular_vector is None:
        angular_vector = np.array([0.0, 0.0, 1.0])
    
    n = vectors.shape[0]
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    angular_vector_norm = angular_vector[0]**2 + angular_vector[1]**2 + angular_vector[2]**2
    angular_vector_norm = math.sqrt(angular_vector_norm)
    if angular_vector_norm == 0:
        raise ValueError("Angular vector cannot be zero vector.")
    angular_vector = angular_vector / angular_vector_norm  # Normalize the angular vector
    ux, uy, uz = angular_vector
    rotated_vectors = np.empty_like(vectors)
    for i in prange(n):
        x, y, z = vectors[i]
        # Apply Rodrigues' rotation formula
        rotated_vectors[i, 0], rotated_vectors[i, 1], rotated_vectors[i, 2] = _apply_rodrigues_formula(
            x, y, z, cos_angle, sin_angle, ux, uy, uz
        )
    # Return the rotated vectors
    return rotated_vectors


@njit(parallel=True, cache=True)
def rotate_deg_in_vector_element_wise(vectors: np.ndarray, angles_degs: np.ndarray, angular_vectors: np.ndarray) -> np.ndarray:
    '''
    Rotate vectors by given angles around specified angular vectors element-wise.
    
    Each vector is rotated by its own angle around its own axis.
    
    Parameters
    ----------
    vectors : np.ndarray
        Array of shape (n, 3) containing vectors to be rotated.
    angles_degs : np.ndarray
        Array of shape (n,) containing angles in degrees by which to rotate the vectors.
    angular_vectors : np.ndarray
        Array of shape (n, 3) where each row is the rotation axis for the corresponding vector.

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) containing the rotated vectors.
        
    Examples
    --------
    >>> import numpy as np
    >>> vectors = np.array([[1, 0, 0], [0, 1, 0]])
    >>> angles = np.array([90.0, 180.0])
    >>> axes = np.array([[0, 0, 1], [0, 0, 1]])  # Both around z-axis
    >>> rotated = rotate_deg_in_vector_element_wise(vectors, angles, axes)
    '''
    n = vectors.shape[0]
    rotated_vectors = np.empty_like(vectors)

    for i in prange(n):
        x, y, z = vectors[i]
        cos_angle = np.cos(np.deg2rad(angles_degs[i]))
        sin_angle = np.sin(np.deg2rad(angles_degs[i]))
        angular_vector_norm = angular_vectors[i, 0]**2 + angular_vectors[i, 1]**2 + angular_vectors[i, 2]**2
        angular_vector_norm = math.sqrt(angular_vector_norm)
        ux, uy, uz = angular_vectors[i] / angular_vector_norm  # Normalize the angular vector
        rotated_vectors[i, 0], rotated_vectors[i, 1], rotated_vectors[i, 2] = _apply_rodrigues_formula(
            x, y, z, cos_angle, sin_angle, ux, uy, uz
        )
    return rotated_vectors


@njit(parallel=True, cache=True)
def rotation_matmul(A: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """
    Perform parallel matrix multiplication of an (k x 3) matrix A with a (3 x 3) rotation matrix.

    This is optimized for satellite position/velocity transformations where
    the same rotation matrix is applied to many 3D vectors.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (k, 3)
    rot : np.ndarray
        Rotation matrix of shape (3, 3)

    Returns
    -------
    np.ndarray
        Result matrix of shape (k, 3) where each row is the rotated vector.
        
    Examples
    --------
    >>> import numpy as np
    >>> positions = np.random.randn(100, 3)  # 100 3D positions
    >>> # Identity rotation
    >>> R = np.eye(3)
    >>> rotated = rotation_matmul(positions, R)
    """
    k = A.shape[0]
    result = np.empty((k, 3), dtype=A.dtype)
    
    for i in prange(k):
        for j in range(3):
            result[i, j] = (A[i, 0] * rot[0, j] + 
                           A[i, 1] * rot[1, j] + 
                           A[i, 2] * rot[2, j])
    
    return result
