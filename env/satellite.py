

import math
import numpy as np

class LEOSatelliteSimulator:
    def __init__(self, name, altitude, inclination, raan, earth_radius=6371):
        self.name = name
        self.altitude = altitude  # km above Earth's surface
        self.inclination = math.radians(inclination)  # Convert to radians
        self.raan = math.radians(raan)  # Right Ascension of the Ascending Node in radians
        self.earth_radius = earth_radius  # Earth's radius in km
        self.semi_major_axis = self.altitude + self.earth_radius  # Semi-major axis in km
        self.mu = 398600  # Earth's gravitational parameter, km^3/s^2
        self.angular_velocity_earth = 7.2921159e-5  # Earth's angular velocity in rad/s

    def orbital_period(self):
        """ Calculate the orbital period of the satellite in minutes """
        T = 2 * math.pi * math.sqrt(self.semi_major_axis**3 / self.mu)  # Period in seconds
        return T / 60  # Convert to minutes

    def position_in_orbit(self, time_elapsed):
        """ Calculate the satellite's position in its orbital plane """
        n = math.sqrt(self.mu / self.semi_major_axis**3)  # Mean motion (rad/s)
        M = n * time_elapsed  # Mean anomaly
        E = M  # For circular orbits (eccentricity e ~ 0), E ≈ M
        true_anomaly = 2 * math.atan2(math.sqrt(1) * math.sin(E/2), math.cos(E/2))  # True anomaly

        # Orbital radius (for circular orbit, it's constant)
        r = self.semi_major_axis

        # Position in orbital plane (ignoring eccentricity for simplicity)
        x_orbit = r * math.cos(true_anomaly)
        y_orbit = r * math.sin(true_anomaly)
        return x_orbit, y_orbit

    def ground_track(self, time_elapsed):
        """ Calculate the satellite's ground track (latitude and longitude) on Earth """
        x_orbit, y_orbit = self.position_in_orbit(time_elapsed)

        # Rotate by the RAAN and Inclination
        x_earth = x_orbit * math.cos(self.raan) - y_orbit * math.sin(self.raan) * math.cos(self.inclination)
        y_earth = y_orbit * math.cos(self.inclination)
        z_earth = y_orbit * math.sin(self.inclination)

        # Longitude of the satellite (considering Earth's rotation)
        lon = math.atan2(z_earth, x_earth) - self.angular_velocity_earth * time_elapsed
        lon = math.degrees(lon) % 360

        # Latitude of the satellite
        lat = math.degrees(math.asin(y_earth / self.semi_major_axis))
        
        # Normalize longitude to [-180, 180]
        if lon > 180:
            lon -= 360

        return lat, lon

    def simulate(self, total_time, time_step):
        """ Simulate the ground track of the satellite over a given time """
        positions = []
        for t in np.arange(0, total_time, time_step):
            lat, lon = self.ground_track(t)
            positions.append((lat, lon))
        return positions

    def __str__(self):
        return f"LEO Satellite '{self.name}' with altitude {self.altitude} km and inclination {math.degrees(self.inclination)}°"

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Use the LEOSatelliteSimulator class defined earlier
    satellite = LEOSatelliteSimulator("Starlink-1", altitude=550, inclination=53, raan=0)

    # Simulate the ground track for one orbit (approx. 90 minutes) with 1-minute time steps
    positions = satellite.simulate(total_time=90*60, time_step=60)

    # Extract the latitudes and longitudes
    lats, lons = zip(*positions)

    # Convert lat/lon to 3D Cartesian coordinates
    earth_radius = 6371  # Earth's radius in km
    x = earth_radius * np.cos(np.radians(lats)) * np.cos(np.radians(lons))
    y = earth_radius * np.cos(np.radians(lats)) * np.sin(np.radians(lons))
    z = earth_radius * np.sin(np.radians(lats))

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth as a 3D sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x_sphere, y_sphere, z_sphere, rstride=4, cstride=4, color='lightblue', alpha=0.5)

    # Use Cartopy to get coastlines and plot them on the 3D sphere
    ax.set_box_aspect([1, 1, 1])

    ax.plot(x, y, z, color='red', marker='o', markersize=4, label=satellite.name)

    # Get coastlines using Cartopy and plot them in 3D
    coastline = cfeature.COASTLINE.with_scale('110m')
    for geom in coastline.geometries():
        if geom.geom_type == 'LineString':
            lons, lats = geom.xy
            x_coast = earth_radius * np.cos(np.radians(lats)) * np.cos(np.radians(lons))
            y_coast = earth_radius * np.cos(np.radians(lats)) * np.sin(np.radians(lons))
            z_coast = earth_radius * np.sin(np.radians(lats))
            ax.plot(x_coast, y_coast, z_coast, color='black')

    # Labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(f"3D Trajectory of {satellite.name} around Earth")

    plt.legend()
    plt.show()