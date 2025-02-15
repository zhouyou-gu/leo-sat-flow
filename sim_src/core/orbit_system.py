from sim_src.core.env_object import EnvObjectRunnable, EnvObject
            
class orbit(EnvObjectRunnable):
    def __init__(self, name, env, orbit_radius, orbital_period, update_interval) -> None:
        super().__init__()
        self.name = name
        self.env = env
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
        
    def run(self):
        """
        SimPy process that updates the satellite's position every update_interval seconds.
        """
        while True:
            # Compute new (x, y) position based on circular orbit.
            x = self.orbit_radius * math.cos(self.theta)
            y = self.orbit_radius * math.sin(self.theta)
            self.body.pos = vector(x, y, 0)  # Orbit in the xy-plane
            print(f"Time {self.env.now:6.2f}: {self.name} at (x={x:7.2f}, y={y:7.2f}), theta={self.theta:6.2f}")
            # Wait for the update interval.
            yield self.env.timeout(self.update_interval)
            # Update the angle.
            self.theta += self.angular_velocity * self.update_interval
            # Keep theta within [0, 2π).
            self.theta %= (2 * math.pi)
            
            
            
class orbit_helper(EnvObject):
    ORBIT_UPDATE_INTERVAL_US = 1e6  # Update interval in microseconds
    SCENE_UPDATE_INTERVAL_US = 1e6  # Update interval in microseconds
    def __init__(self, earth, n_satellite) -> None:
        super().__init__()
        self.earth = earth
        self.orbit_list = []

    def get_orbits(self, n_satellite):
        for i in range(n_satellite):
            orbit = orbit(f"Satellite {i}", self.env, 10, 10, 1)
            
    def visualize(self):
        self.setup_scene()
        while True:
            rate(30)       
            
                          
    def setup_scene():
        """
        Set up the VPython 3D scene with a textured Earth.
        The Earth is centered at the origin with a radius of 1 simulation unit.
        """
        scene.width = 800
        scene.height = 600
        scene.background = color.white
        scene.lights = []

        # Create a VPython sphere to represent Earth at the origin.
        earth = sphere(pos=vector(0, 0, 0), radius=1,
                       texture="simple_earth_texture.png",emissive=True)
        # earth = sphere(pos=vector(0, 0, 0), radius=1,
        #                texture=textures.earth,
        #                shininess=0.8)
        return earth


    