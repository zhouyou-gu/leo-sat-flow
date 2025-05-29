import pandas as pd
import numpy as np
from sim_mld.constellation import *


class terrain:
    def __init__(self):
        self.earth_rotation_deg = 0.0
        
        self.ground_station_positions = self._load_ground_station_positions()
        self.ground_station_positions_rotated = rotate_deg_in_vector(self.ground_station_positions, self.earth_rotation_deg)

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
        
    def get_ground_station_positions(self):
        """
        Get the ground station positions in Cartesian coordinates.
        
        Returns:
            np.ndarray: Ground station positions in Cartesian coordinates.
        """
        return self.ground_station_positions_rotated

    def get_traffic_info(self, positions, seed=0):
        pass



if __name__ == "__main__":
    pass
    