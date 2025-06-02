import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from numba import njit
from sim_mld.constellation import *


from numba import njit, prange

@njit(parallel=True, cache=True)
def all_one_pairs_parallel(arr1, arr2, n1, n2):
    """
    Given two 1D NumPy arrays (arr1, arr2) of 0s and 1s, returns
    a NumPy array of shape (n1*n2, 2) containing all index pairs (i, j)
    such that arr1[i] != 0 and arr2[j] != 0.
    """

    # 3) Collect indices of non-zero entries (serial)
    idx1 = np.empty(n1, np.int64)
    idx2 = np.empty(n2, np.int64)

    count = 0
    for i in range(arr1.shape[0]):
        if arr1[i] != 0:
            idx1[count] = i
            count += 1

    count = 0
    for j in range(arr2.shape[0]):
        if arr2[j] != 0:
            idx2[count] = j
            count += 1

    # 4) Allocate output array
    total_pairs = n1 * n2
    output = np.empty((total_pairs, 2), np.int64)

    # 5) Fill in all pairs in parallel:
    #    We let 'a' run in parallel, and compute pos = a * n2 + b
    for a in prange(n1):
        for b in range(n2):
            pos = a * n2 + b
            output[pos, 0] = idx1[a]
            output[pos, 1] = idx2[b]

    return output

@njit(parallel=True,cache=True)
def query_user_distribution(sat_lat_lon_positions, user_distribution_repeated, cell_size_in_idx):
    """
    Query the user distribution for the given satellite positions.
    
    Parameters:
        sat_positions (np.ndarray): Satellite positions in Cartesian coordinates.
        user_distribution (np.ndarray): User distribution data.
        seed (int): Random seed for reproducibility.
        
    Returns:
        np.ndarray: User distribution values at the satellite positions.
    """
    n_sat = sat_lat_lon_positions.shape[0]
    user_values = np.zeros(sat_lat_lon_positions.shape[0])
    shape_x = user_distribution_repeated.shape[0]
    shape_y = user_distribution_repeated.shape[1]
    
    centre_x = shape_x // 2
    centre_y = shape_y // 2
    
    for i in prange(n_sat):
        lat, lon = sat_lat_lon_positions[i]
        lat_idx = int((lat + np.pi / 2) / (np.pi / shape_x)) + centre_x
        lon_idx = int((lon + np.pi) / (2 * np.pi / shape_y)) + centre_y

        # get the box bounds around the lat/lon
        # and sum the values in the box
        # This is a simple nearest neighbor approach, could be improved with interpolation
        # sum the values in the tile box
        lat_start = max(0, lat_idx - cell_size_in_idx // 2)
        lat_end = min(shape_x, lat_idx + cell_size_in_idx // 2 + 1)
        lon_start = max(0, lon_idx - cell_size_in_idx // 2)
        lon_end = min(shape_y, lon_idx + cell_size_in_idx // 2 + 1)
        for j in range(lat_start, lat_end):
            for k in range(lon_start, lon_end):
                user_values[i] += user_distribution_repeated[j, k]
    return user_values


class terrain:
    EARTH_RADIUS = 6371.0  # in kilometers
    EQUATOR_CIRCUMFERENCE = 2 * np.pi * EARTH_RADIUS  # in kilometers
    CELL_SIZE = 200  # in kilometers, size of the cell for user distribution
    ACTIVE_USER_PERCENTAGE = 1e-4 # percentage of active users
    SOURCE_DL_RATE = 0.1  # traffic source rate in Gbps
    SOURCE_UL_RATE = 0.05  # traffic source rate in Gbps
    TARGET_DL_RATE = 20 # target downlink rate in Gbps
    TARGET_UL_RATE = 20  # target uplink rate in Gbps
    GW_RANGE = 600.0  # in kilometers, range of the ground station
    
    def __init__(self):
        self.earth_rotation_deg = 0.0
        
        self.ground_station_positions = self._load_ground_station_positions()
        self.ground_station_positions_rotated = None
        self.ground_station_positions_rotated_kd_tree = None
        self.update_earth_rotation(self.earth_rotation_deg)

        self.user_distribution_repeated = self._load_user_distribution()

    def _load_user_distribution(self):
        npy_path = "population_density_texture.npy"
        user_distribution = np.load(npy_path)
        user_distribution_repeated = np.concatenate((user_distribution, user_distribution), axis=1)
        return user_distribution_repeated

    def _load_ground_station_positions(self):
        """
        Initialize the ground station positions from a CSV file.
        The CSV should contain latitude and longitude in degrees.
        """
        csv_path = "satnogs_locations.csv"
        df = pd.read_csv(csv_path)
        ground_station_positions = df[['lat', 'lng']].to_numpy()
        ground_station_positions = np.deg2rad(ground_station_positions)
        ground_station_positions = lat_lon_to_xyz(ground_station_positions)
        
        return ground_station_positions

    def update_earth_rotation(self, earth_rotation_deg):
        """
        Update the earth rotation based on the given timestamp.
        
        Parameters:
            ts: Timestamp to update the earth rotation.
        """
        self.earth_rotation_deg = earth_rotation_deg
        self.ground_station_positions_rotated = rotate_deg_in_vector(self.ground_station_positions, self.earth_rotation_deg)
        self.ground_station_positions_rotated_kd_tree = cKDTree(self.ground_station_positions_rotated)
    
    def get_ground_station_positions(self):
        """
        Get the ground station positions in Cartesian coordinates.
        
        Returns:
            np.ndarray: Ground station positions in Cartesian coordinates.
        """
        return self.ground_station_positions_rotated

    def get_traffic_info(self, sat_positions, seed=0):
        rng = np.random.default_rng(seed)
        n_sat = sat_positions.shape[0]
        n_pair = 1000
        rng = np.random.default_rng(seed)
        
        sources = rng.integers(0, n_sat, n_pair)
        targets = rng.integers(0, n_sat, n_pair)
        while np.any(sources == targets):
            mask = (sources == targets)
            targets[mask] = rng.integers(0, n_sat, np.sum(mask))
        
        data_source, data_target = sources, targets
        
        return data_source, data_target

    def get_traffic_info_test(self, sat_positions, seed=0):
        sat_lat_lon_positions = xyz_to_lat_lon(sat_positions)
        cell_size_in_idx = int(self.CELL_SIZE / self.EQUATOR_CIRCUMFERENCE * self.user_distribution_repeated.shape[1]/2.)

        user_distribution_values = query_user_distribution(sat_lat_lon_positions, self.user_distribution_repeated, cell_size_in_idx)
        user_distribution_values += self.CELL_SIZE ** 2
        print("user_distribution_values:", user_distribution_values.mean(), user_distribution_values.max(), user_distribution_values.min())

        rng = np.random.default_rng(seed)
        # exponential distribution for user traffic
        user_per_sat = rng.poisson(lam=user_distribution_values)
        
        active_users = user_per_sat * self.ACTIVE_USER_PERCENTAGE

        traffic_dl_rate_source = active_users * self.SOURCE_DL_RATE
        traffic_ul_rate_source = active_users * self.SOURCE_UL_RATE

        # find the ground stations that are within range of the satellite positions
        distance, index = self.ground_station_positions_rotated_kd_tree.query(sat_positions, distance_upper_bound=self.GW_RANGE / self.EARTH_RADIUS, workers=-1)

        connected_to_ground_stations = index != self.ground_station_positions_rotated_kd_tree.n
        traffic_dl_rate_target = connected_to_ground_stations.astype(float) * self.TARGET_DL_RATE
        traffic_ul_rate_target = connected_to_ground_stations.astype(float) * self.TARGET_UL_RATE

        print("connected_to_ground_stations:", connected_to_ground_stations.sum())

        print("traffic_dl_rate_source:", traffic_dl_rate_source)
        forward_traffic_capacity = np.clip(traffic_ul_rate_target - traffic_dl_rate_source, 0, None)
        reverse_traffic_capacity = np.clip(traffic_dl_rate_target - traffic_ul_rate_source, 0, None)

        forward_traffic_demand = np.clip(traffic_dl_rate_source - traffic_ul_rate_target, 0, None)
        reverse_traffic_demand = np.clip(traffic_ul_rate_source - traffic_dl_rate_target, 0, None)

        pairs = all_one_pairs_parallel(forward_traffic_capacity, forward_traffic_demand, np.count_nonzero(forward_traffic_capacity), np.count_nonzero(forward_traffic_demand))
        print("pairs shape:", pairs.shape)
        return pairs[:, 0], pairs[:, 1], forward_traffic_capacity, forward_traffic_demand

if __name__ == "__main__":
    pass
    