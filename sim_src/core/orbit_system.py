import math
import time
import numpy as np
from numba import njit
from sim_src.core.earth_system import earth
from sim_src.core.env_object import EnvObjectRunnable, EnvObject
from sim_src.core.visual_system import VisObject
from vpython import sphere, vector, color, arrow


@njit
def rotate_vector(pos, L, omega, t_us):
    # Normalize the angular momentum vector
    t = t_us/1e6
    n = L / np.linalg.norm(L)
    theta = omega * t

    # Compute Rodrigues' rotation
    pos_new = (pos * np.cos(theta) +
            np.cross(n, pos) * np.sin(theta) +
            n * np.dot(n, pos) * (1 - np.cos(theta)))
    return pos_new

class orbit(EnvObjectRunnable,VisObject):
    orbit_update_interval_us = 1000000  # Update interval in microseconds
    def __init__(self, id=0, altitude=350, inclination_deg=0, raan_deg=0, init_period_offset_pct = 0.1, earth_radius=6371):
        super().__init__()
        self.id = id
        self.altitude = altitude
        self.inclination_deg = inclination_deg
        self.raan_deg = raan_deg
        self.init_period_offset_pct = init_period_offset_pct
        self.radius = (self.altitude + earth_radius)/earth_radius
        # self.semi_major_axis = self.altitude + earth_radius
        self.period_us = self.orbital_period_us(self.altitude)
        print(self.period_us/1e6/60)
        self.angular_speed = 2*math.pi/(self.period_us/1e6)
        self.angular_momentum_vector = self.orbital_plane_normal(self.inclination_deg, self.raan_deg)
        v_arrow = arrow(pos=vector(0, 0, 0), axis=vector(*self.angular_momentum_vector)*1.2, color=color.black,shaftwidth = 0.01)
        self.pos = np.array([self.radius * math.sin(np.radians(self.raan_deg)), 0, self.radius * math.cos(np.radians(self.raan_deg))])
        ## TODO: update the position of the satellite based on the period offset
        self.pos = rotate_vector(self.pos,self.angular_momentum_vector,self.angular_speed, self.period_us * self.init_period_offset_pct)
    
    def run(self):
        """
        SimPy process that updates the satellite's position every update_interval seconds.
        """
        while True:
            yield self.env.timeout(self.orbit_update_interval_us)
            self.pos = rotate_vector(self.pos,self.angular_momentum_vector,self.angular_speed, self.orbit_update_interval_us)
            # self.init_period_offset_pct += self.update_interval_us / self.period_us

    def upd_vis_object(self):
        if not self.plot_obj:
            self.plot_obj = sphere(pos=vector(*self.pos), radius=0.01,
                    color=color.blue,emissive=True)
            return self.plot_obj
        else:
            self.plot_obj.pos = vector(*self.pos)
            
    @staticmethod
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

    @staticmethod
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
    
    



if __name__ == "__main__":
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
    # x_arrow = arrow(pos=vector(0, 0, 0), axis=vector(1,0,0)*1.2, color=color.blue,shaftwidth = 0.01)
    # y_arrow = arrow(pos=vector(0, 0, 0), axis=vector(0,1,0)*1.2, color=color.green,shaftwidth = 0.01)
    # z_arrow = arrow(pos=vector(0, 0, 0), axis=vector(0,0,1)*1.2, color=color.yellow,shaftwidth = 0.01)
    # zx_ar = arrow(pos=vector(0, 0, 0), axis=vector(1,0,1)*1.2, color=color.yellow,shaftwidth = 0.01)
    for i in range(1000):
        # to = orbit(inclination_deg=45,raan_deg=45,init_period_offset_pct=0.5)   
        to = orbit(altitude=np.random.rand()*10000+350, inclination_deg=np.random.rand()*45,raan_deg=np.random.rand()*360,init_period_offset_pct=np.random.rand())   
    vs = visual_system()
    EnvObject.run(until=2000000000000)
    