#!/usr/bin/env python3
"""
Satellite Orbit Simulation with 3D Earth Visualization

This simulation uses SimPy's real-time environment (simpy.rt.RealtimeEnvironment)
to update satellite positions and VPython to visualize a 3D Earth (with a terrestrial texture)
and the satellites moving in circular orbits.

Note:
  - The simulation uses a simplified orbit model (circular, in the xy-plane).
  - The Earth is represented by a sphere at the origin with a built-in Earth texture.
  - The simulation time is synchronized to wall-clock time (1 simulation second = 1 real second).
"""

import math
import simpy.rt
from vpython import sphere, vector, textures, color, scene, rate

class Satellite:
    """
    A satellite in a circular orbit that periodically updates its position.

    Attributes:
        env (simpy.rt.RealtimeEnvironment): The simulation environment.
        name (str): Satellite name.
        orbit_radius (float): Radius of the orbit (in simulation units).
        orbital_period (float): Time (in seconds) to complete one full orbit.
        update_interval (float): Time interval (in seconds) between position updates.
        angular_velocity (float): Angular velocity in radians/second.
        theta (float): Current angular position in radians.
        body (vpython.sphere): VPython object representing the satellite.
    """
    def __init__(self, env: simpy.rt.RealtimeEnvironment, name: str,
                 orbit_radius: float, orbital_period: float, update_interval: float):
        self.env = env
        self.name = name
        self.orbit_radius = orbit_radius
        self.orbital_period = orbital_period
        self.update_interval = update_interval
        # Calculate angular velocity: ω = 2π / period
        self.angular_velocity = 2 * math.pi / orbital_period
        self.theta = 0.0  # Start at angle 0 radians

        # Create a VPython sphere to represent the satellite.
        # The satellite size is set relative to its orbit for visibility.
        self.body = sphere(pos=vector(orbit_radius, 0, 0),
                           radius=orbit_radius * 0.05,
                           color=color.red, make_trail=True,
                           trail_type="curve", retain=50)
        # Start the orbit process.
        self.action = env.process(self.orbit())

    def orbit(self):
        """
        SimPy process that updates the satellite's position every update_interval seconds.
        """
        while True:
            # Compute new (x, y) position based on circular orbit.
            x = self.orbit_radius * math.cos(self.theta)
            y = self.orbit_radius * math.sin(self.theta)
            self.body.pos = vector(x, y, 0)  # Orbit in the xy-plane
            self.body.color = color.red
            if scene.range < 2:
                scene.range = 2
            print(f"Time {self.env.now:6.2f}: {self.name} at (x={x:7.2f}, y={y:7.2f}), theta={self.theta:6.2f}")
            # Wait for the update interval.
            yield self.env.timeout(self.update_interval)
            # Update the angle.
            self.theta += self.angular_velocity * self.update_interval
            # Keep theta within [0, 2π).
            self.theta %= (2 * math.pi)

def setup_scene():
    """
    Set up the VPython 3D scene with a textured Earth.
    The Earth is centered at the origin with a radius of 1 simulation unit.
    """
    # scene.width = 800
    # scene.height = 600
    # scene.background = color.white
    # scene.lights = []

    # Create a VPython sphere to represent Earth at the origin.
    earth = sphere(pos=vector(0, 0, 0), radius=1,
                   texture="simple_earth_texture.png",emissive=True)
    # earth = sphere(pos=vector(0, 0, 0), radius=1,
    #                texture=textures.earth,
    #                shininess=0.8)
    return earth

def main():
    """
    Create the simulation environment, set up the 3D scene and satellites,
    and run the simulation.
    """
    # Initialize the VPython scene.
    setup_scene()

    # Create a real-time simulation environment (1 simulation sec = 1 real sec).
    env = simpy.rt.RealtimeEnvironment(factor=1, strict=False)

    # For visualization purposes, we scale the Earth to radius=1.
    # Satellite orbits are defined relative to this scale.
    # Example: Satellite-1 orbits at 1.5 units; Satellite-2 at 2.0 units.
    sat1 = Satellite(env, "Satellite-1", orbit_radius=1.5, orbital_period=20, update_interval=0.01)
    sat2 = Satellite(env, "Satellite-2", orbit_radius=2.0, orbital_period=30, update_interval=0.01)

    # Run the simulation for 60 seconds.
    env.run(until=100)

if __name__ == "__main__":
    main()
    from vpython.no_notebook import stop_server
    stop_server()