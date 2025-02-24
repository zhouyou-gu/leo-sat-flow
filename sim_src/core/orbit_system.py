import math
import time
import numpy as np
from numba import njit, jit
from sim_src.core.earth_system import earth
from sim_src.core.env_object import EnvObjectRunnable, EnvObject
from sim_src.core.visual_system import VisObject
from vpython import sphere, vector, color, arrow, cylinder
from scipy.spatial import cKDTree
import hnswlib


@njit
def rotate_a_vector(pos, L, omega, t_us):
    # Normalize the angular momentum vector
    t = t_us/1e6
    n = L / np.linalg.norm(L)
    theta = omega * t

    # Compute Rodrigues' rotation
    pos_new = (pos * np.cos(theta) +
            np.cross(n, pos) * np.sin(theta) +
            n * np.dot(n, pos) * (1 - np.cos(theta)))
    return pos_new

def rotate_vectors(pos, L, omega, t_us):
    # Convert time from microseconds to seconds; t has shape (K, 1)
    t = t_us / 1e6
    # Compute rotation angle theta for each row; theta has shape (K, 1)
    theta = omega * t

    # Normalize the angular momentum vector row-wise.
    # Compute the norm of each row in L and keep dimensions for broadcasting.
    L_norm = np.sqrt(np.sum(L**2, axis=1, keepdims=True))
    n = L / L_norm  # n has shape (K, 3)

    # Compute cosine and sine of theta for each row; these will be (K, 1)
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    # Compute the dot product of n and pos for each row (result is (K, 1))
    dot_n_pos = np.sum(n * pos, axis=1, keepdims=True)
    
    # Compute the cross product row-wise; result is (K, 3)
    cross_n_pos = np.cross(n, pos)
    
    # Apply Rodrigues' rotation formula row-wise:
    # pos_new = pos * cos(theta) +
    #           (n x pos) * sin(theta) +
    #           n * (n . pos) * (1 - cos(theta))
    pos_new = pos * cosTheta + cross_n_pos * sinTheta + n * dot_n_pos * (1 - cosTheta)
    return pos_new

@njit
def orbital_period_us(altitude_km):
    # Constants
    mu = 3.986004418e14         # Earth's gravitational parameter in m^3/s^2
    earth_radius = 6371e3         # Earth's radius in meters

    # Convert altitude from kilometers to meters
    altitude_m = altitude_km * 1000

    # Calculate the semi-major axis (a)
    a = earth_radius + altitude_m

    # Compute the orbital period T using Kepler's Third Law
    T = 2 * math.pi * math.sqrt(a**3 / mu)

    return T * 1e6 # Convert to microseconds

@njit
def orbital_plane_normal(inclination_deg, raan_deg):
    # Convert degrees to radians
    i = np.radians(inclination_deg)
    Omega = np.radians(raan_deg)
    # Compute the unit normal vector (angular momentum vector)
    # Using the convention: h = [ -sin(i)*sin(Omega), sin(i)*cos(Omega), cos(i) ]
    h = np.array([
        np.sin(i) * np.cos(Omega),
        np.cos(i),
        -np.sin(i) * np.sin(Omega),
    ])
    # Normalize (should already be unit length but ensures robustness)
    h_normalized = h / np.linalg.norm(h)
    return h_normalized


def find_nearest_neighbor(target_vector, tree, k=2):
    """
    Finds the nearest neighbor (by Euclidean distance) of target_vector in row_vectors
    using a KD-Tree for efficient querying.
    Parameters:
        target_vector (np.ndarray): 1D array representing the target vector.
        row_vectors (np.ndarray): 2D array where each row is a vector.
    Returns:
        nearest_index (int): Index of the nearest neighbor in row_vectors.
        nearest_distance (float): Euclidean distance to the nearest neighbor.
    """
    # Build the KD-tree from the row vectors
    
    # Query the tree for the nearest neighbor
    nearest_distance, nearest_index = tree.query(target_vector,k=2)
    
    return nearest_index, nearest_distance


# class orbit(EnvObjectRunnable,VisObject):
#     orbit_update_interval_us = 1000000  # Update interval in microseconds
#     def __init__(self, id=0, altitude=350, inclination_deg=0, raan_deg=0, init_period_offset_pct = 0.1, earth_radius=6371):
#         super().__init__()
#         self.id = id
#         self.altitude = altitude
#         self.inclination_deg = inclination_deg
#         self.raan_deg = raan_deg
#         self.init_period_offset_pct = init_period_offset_pct
#         self.radius = (self.altitude + earth_radius)/earth_radius
#         # self.semi_major_axis = self.altitude + earth_radius
#         self.period_us = self.orbital_period_us(self.altitude)
#         print(self.period_us/1e6/60)
#         self.angular_speed = 2*math.pi/(self.period_us/1e6)
#         self.angular_momentum_vector = self.orbital_plane_normal(self.inclination_deg, self.raan_deg)
#         v_arrow = arrow(pos=vector(0, 0, 0), axis=vector(*self.angular_momentum_vector)*1.2, color=color.black,shaftwidth = 0.01)
#         self.pos = np.array([self.radius * math.sin(np.radians(self.raan_deg)), 0, self.radius * math.cos(np.radians(self.raan_deg))])
#         ## TODO: update the position of the satellite based on the period offset
#         self.pos = rotate_a_vector(self.pos,self.angular_momentum_vector,self.angular_speed, self.period_us * self.init_period_offset_pct)
    
#     def run(self):
#         """
#         SimPy process that updates the satellite's position every update_interval seconds.
#         """
#         while True:
#             yield self.env.timeout(self.orbit_update_interval_us)
#             self.pos = rotate_a_vector(self.pos,self.angular_momentum_vector,self.angular_speed, self.orbit_update_interval_us)
#             # self.init_period_offset_pct += self.update_interval_us / self.period_us

#     def upd_vis_object(self):
#         if not self.plot_obj:
#             self.plot_obj = sphere(pos=vector(*self.pos), radius=0.01,
#                     color=color.blue,emissive=True)
#             return self.plot_obj
#         else:
#             self.plot_obj.pos = vector(*self.pos)

class orbit_system(EnvObjectRunnable,VisObject):
    orbit_update_interval_us = 1000000  # Update interval in microseconds
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.pos = np.empty((0,3))
        self.amv = np.empty((0,3))
        self.period_us = np.empty((0,1))
        self.angspd_rs = np.empty((0,1))
        self.orb_group = []
 
        self.tree = cKDTree(np.empty((0, 3)))
        
    def add_orbit(self, orbit_group=0, altitude=0, inclination_deg=0, raan_deg=0, init_period_offset_pct = 0.1, earth_radius=6371):        
        radius = (altitude + earth_radius)/earth_radius
        period_us = orbital_period_us(altitude)
        angular_speed = 2*math.pi/(period_us/1e6)
        angular_momentum_vector = orbital_plane_normal(inclination_deg, raan_deg)
        
        self.period_us = np.vstack([self.period_us, period_us])
        self.angspd_rs = np.vstack([self.angspd_rs, angular_speed])
        self.orb_group.append(orbit_group)
        
        # v_arrow = arrow(pos=vector(0, 0, 0), axis=vector(*angular_momentum_vector)*1.2, color=color.black,shaftwidth = 0.01)
        pos = np.array([radius * math.sin(np.radians(raan_deg)), 0, radius * math.cos(np.radians(raan_deg))])
        ## TODO: update the position of the satellite based on the period offset
        pos = rotate_a_vector(pos, angular_momentum_vector, angular_speed, period_us * init_period_offset_pct)
    
        self.pos = np.vstack([self.pos, pos])
        self.amv = np.vstack([self.amv, angular_momentum_vector])
        
        self.counter += 1
        return self.counter-1
    
    def add_orbits(self, altitude=350, inclination_deg= 180, raan_deg=180,  n=100):
        ret = []
        for i in range(n):
            init_period_offset_pct = np.random.uniform(0, 1)
            idx = self.add_orbit(altitude=altitude, inclination_deg=inclination_deg, raan_deg=raan_deg, init_period_offset_pct=init_period_offset_pct)
            ret.append(idx)
        return ret
    
    def add_rand_orbits(self, n=100):
        ret = []
        altitude = 350
        inclination_deg = np.random.uniform(0, 90)
        raan_deg = np.random.uniform(-180, 180)
        for i in range(n):
            init_period_offset_pct = np.random.uniform(0, 1)
            idx = self.add_orbit(altitude=altitude, inclination_deg=inclination_deg, raan_deg=raan_deg, init_period_offset_pct=init_period_offset_pct)
            ret.append(idx)
        return ret

    def add_unif_orbits(self, raan_deg):
        ret = []
        altitude = 350
        inclination_deg = 50
        # raan_deg = np.arange(-180, 180, 15)
        init_period_offset_pct_list = np.arange(0, 1, 0.02).tolist()
        for i in init_period_offset_pct_list:
            init_period_offset_pct = i
            idx = self.add_orbit(altitude=altitude, inclination_deg=inclination_deg, raan_deg=raan_deg, init_period_offset_pct=init_period_offset_pct)
            ret.append(idx)
        return ret
    
    def upd_vis_object(self):
        if not self.plot_obj:
            self.plot_obj = []
            for i in range(self.counter):
                self.plot_obj.append(sphere(pos=vector(*self.pos[i]), radius=0.01,
                    color=color.blue,emissive=True))
            self.links = []
            for i in range(self.counter):
                j, _ = find_nearest_neighbor(self.pos[i],self.tree)
                j = j[-1]
                link = cylinder(pos=self.plot_obj[i].pos, 
                        axis=(self.plot_obj[j].pos - self.plot_obj[i].pos), 
                        radius=0.005, 
                        color=color.red)
                self.links.append(link)
        
        else:
            for i in range(self.counter):
                self.plot_obj[i].pos = vector(*self.pos[i])
                j, _ = find_nearest_neighbor(self.pos[i],self.tree)
                j = j[-1]
                self.links[i].pos = self.plot_obj[i].pos
                self.links[i].axis = self.plot_obj[j].pos - self.plot_obj[i].pos
                
    def run(self):
        while True:
            yield self.env.timeout(self.orbit_update_interval_us)
            self.pos = rotate_vectors(self.pos, self.amv, self.angspd_rs,np.ones_like(self.angspd_rs)*self.orbit_update_interval_us)
            tic = time.time()
            self.reset_tree()
            # print(time.time()-tic)
            
    def get_loc(self, idx):
        return self.pos[idx]

    def reset_tree(self):
        self.tree = cKDTree(self.pos)


if __name__ == "__main__":
    from sim_src.core.visual_system import visual_system
    EnvObject.init_env(RT=True,scaling=100)
    # for i in [-1, 0, 1]:
    #     for j in [-1, 0, 1]:
    #         for k in [-1, 0, 1]:
    #             # Skip the zero vector to avoid creating an arrow with no length
    #             if i == 0 and j == 0 and k == 0:
    #                 continue
    #             v = np.array([i,j,k])
    #             c = np.abs(v)
    #             if np.sum(c) >=3:
    #                 continue
    #             v_norm = v/np.linalg.norm(v)
    #             # Create an arrow from the origin in the direction of the vector (i, j, k)
    #             arrow(pos=vector(0, 0, 0),
    #                 axis=vector(*v_norm)*1.2,
    #                 shaftwidth=0.05,
    #                 color=vector(*c))
    

    # for i in range(1000):
    #     # to = orbit(inclination_deg=45,raan_deg=45,init_period_offset_pct=0.5)   
    #     to = orbit(altitude=np.random.rand()*10000+350, inclination_deg=np.random.rand()*45,raan_deg=np.random.rand()*360,init_period_offset_pct=np.random.rand())   
    os = orbit_system()
    for i in range(24):
        os.add_unif_orbits(raan_deg=15*i)
    os.reset_tree()
    et = earth(orbit_system=os)

    vs = visual_system()
    EnvObject.run(until=20000000000)
    