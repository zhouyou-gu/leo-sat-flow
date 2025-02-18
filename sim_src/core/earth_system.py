import math

from numba import njit
from PIL import Image
import numpy as np
from sim_src.core.env_object import EnvObjectRunnable, EnvObject
from sim_src.core.visual_system import VisObject
from vpython import sphere, vector, color, arrow

@njit
def generate_random_lat_lon_np(center_lat, center_lon, angular_radius, num_samples=100):
    """
    Generate random points (lat, lon) uniformly distributed within a spherical cap.
    Parameters:
    - center_lat: Center latitude in radians.
    - center_lon: Center longitude in radians.
    - angular_radius: Angular radius of the cap in radians.
    - num_samples: Number of random points to generate.
    
    Returns:
    - lat: Array of generated latitudes in radians.
    - lon: Array of generated longitudes in radians.
    """
    
    center_lat = np.arcsin(np.sin(center_lat))
    
    # Uniform random numbers for the cumulative area distribution
    u = np.random.rand(num_samples)
    gamma = np.arccos(1 - u * (1 - np.cos(angular_radius)))  # Angular distances for uniform area sampling
    theta = 2 * np.pi * np.random.rand(num_samples)            # Random bearings

    # Handle the polar case separately:
    if np.abs(np.abs(center_lat) - np.pi/2) < 1e-8:
        lat = np.where(center_lat > 0, np.pi/2 - gamma, -np.pi/2 + gamma)
        lon = (center_lon + 2 * np.pi * np.random.rand(num_samples) + np.pi) % (2 * np.pi) - np.pi
    else:
        lat = np.arcsin(np.sin(center_lat) * np.cos(gamma) +
                        np.cos(center_lat) * np.sin(gamma) * np.cos(theta))
        lon = center_lon + np.arctan2(np.sin(theta) * np.sin(gamma) * np.cos(center_lat),
                                      np.cos(gamma) - np.sin(center_lat) * np.sin(lat))
        lon = (lon + np.pi) % (2 * np.pi) - np.pi  # Normalize lon to [-pi, pi]

    return lat, lon

@njit
def lat_lon_to_xyz(lat, lon, r=1.1):
    """
    Convert latitude and longitude arrays (in radians) to 3D Cartesian coordinates on a sphere.
    
    Parameters:
    - lat: Array of latitudes in radians.
    - lon: Array of longitudes in radians.
    - r: Radius of the sphere (default is 1).
    
    Returns:
    - x, y, z: Arrays of Cartesian coordinates.
    """
    x = r * np.cos(lat) * np.sin(lon)
    y = r * np.sin(lat)
    z = r * np.cos(lat) * np.cos(lon)
    return x, y, z

@njit
def xyz_to_lat_lon(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (lat, lon, r).

    Uses the formulas:
        r   = sqrt(x^2 + y^2 + z^2)
        lat = arcsin(y / r)
        lon = arctan2(z, x)
        
    Parameters:
        x, y, z : float or array_like
            Cartesian coordinates.
        
    Returns:
        lat, lon, r : same shape as input arrays
            - lat: latitude in radians (range: [-pi/2, pi/2])
            - lon: longitude in radians (range: [-pi, pi])
            - r  : radius
    """
    r = np.sqrt(x*x + y*y + z*z)
    lat = np.arcsin(y / r)
    lon = np.arctan2(x, z)
    return lat, lon, r

@njit
def lat_lon_to_pixel(lat, lon, image):
    """
    Convert latitude and longitude (in degrees) to pixel coordinates.
    Assumes:
      - Longitude in [-180, 180] maps to x in [0, width]
      - Latitude in [90, -90] maps to y in [0, height]
    """
    height, width = image.shape
    x = np.floor((lon + np.pi) / (2 * np.pi) * width).astype(np.int64)
    y = np.floor(((np.pi / 2) - lat) / np.pi * height).astype(np.int64)
    return y, x



class earth(EnvObjectRunnable,VisObject):
    ROTAION_PERIOD = 86164.0905
    earth_update_interval_us = 1000000
    def __init__(self):
        super().__init__()
        self.angle = 0
        self.land_texture = np.array(Image.open("land_sea_texture_bw.png"))
        self.cities = np.loadtxt("cities.csv",delimiter=",")
        # x, y, z = lat_lon_to_xyz(np.radians(self.cities[:,0]),np.radians(self.cities[:,1])+self.angle)
        # for i in range(100):
        #     plot_obj = sphere(pos=vector(x[i],y[i],z[i]), radius=0.01,color=color.blue)
            
    def run(self):
        while True:
            yield self.env.timeout(self.earth_update_interval_us)
            self.angle += 2*math.pi/self.ROTAION_PERIOD/1e6 * self.earth_update_interval_us

    def upd_vis_object(self):
        if not self.plot_obj:
            self.plot_obj = sphere(pos=vector(0, 0, 0), radius=1,
                texture="simple_earth_texture.png",emissive=True)
            return self.plot_obj
        else:
            new_axis = vector(1, 0, 0).rotate(angle=self.angle, axis=vector(0, 1, 0))
            new_up   = vector(0, 1, 0).rotate(angle=self.angle, axis=vector(0, 1, 0))
            # Set these vectors to the sphere. This sets its orientation absolutely.
            self.plot_obj.axis = new_axis
            self.plot_obj.up   = new_up
            
    def query_random_binary_points(self, latitude, longitude, angle, num_samples=20):
        lat, lon = generate_random_lat_lon_np(np.radians(latitude),np.radians(longitude),np.radians(angle),num_samples=num_samples)
        # x, y, z = lat_lon_to_xyz(lat, lon, r=1.1)
        # for i in range(num_samples):
        #     plot_obj = arrow(pos=vector(0, 0, 0), axis=vector(*(x[i],y[i],z[i]))*1.2, color=color.black,shaftwidth = 0.01)
        #     plot_obj.rotate(angle=self.angle, axis=vector(0, 1, 0))
            
        pix_y, pix_x = lat_lon_to_pixel(lat,lon,self.land_texture)
        pix = self.land_texture[pix_y,pix_x]
        print(np.mean(pix))
   
    def query_random_binary_points_at_abs_cor(self, pos, angle, num_samples=20):
        # plot_obj = sphere(pos=vector(*(pos*1.1)), radius=0.2,color=color.blue)
        pos = np.reshape(pos, (-1, 3))
        lat, lon, r = xyz_to_lat_lon(pos[:,0],pos[:,1],pos[:,2])
        self.query_random_binary_points(np.degrees(lat),np.degrees(lon-self.angle),angle,num_samples)
            
if __name__ == "__main__":
    h = earth.ROTAION_PERIOD/60/60
    print(f"Earth rotates once every {h} hours.")
    
    from sim_src.core.visual_system import visual_system
    EnvObject.init_env(RT=True)
    et = earth()
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                # Skip the zero vector to avoid creating an arrow with no length
                if i == 0 and j == 0 and k == 0:
                    continue
                v = np.array([i,j,k])
                c = np.abs(v)
                if np.sum(c) >=3:
                    continue
                v_norm = v/np.linalg.norm(v)
                # Create an arrow from the origin in the direction of the vector (i, j, k)
                arrow(pos=vector(0, 0, 0),
                    axis=vector(*v_norm)*1.2,
                    shaftwidth=0.05,
                    color=vector(*c))
    
    # et.query_random_binary_points(0,0,15,100)
    et.query_random_binary_points_at_abs_cor(np.array([0,-1,0]),1,100)
    # to = orbit()   
    vs = visual_system()
    EnvObject.run(until=2000000000000)
    