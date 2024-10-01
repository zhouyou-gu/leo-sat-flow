import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, GCRS
from astropy.time import Time

# Define the parameters for the LEO orbit
semi_major_axis = 7000 * u.km  # Semi-major axis for LEO (~7000 km)
eccentricity = 0.001  # Near circular orbit
inclination = 98.6 * u.deg  # Polar orbit inclination
raan = 0 * u.deg  # Right ascension of ascending node
argp = 0 * u.deg  # Argument of perigee
nu = 0 * u.deg  # True anomaly

# Define the initial position and velocity vectors
r = CartesianRepresentation([7000, 0, 0] * u.km)
v = CartesianDifferential([0, 7.5, 0] * u.km/u.s)
state_vector = r.with_differentials(v)

# Define the reference frame
gcrs_frame = GCRS(obstime=Time.now())

# Propagate the orbit over one period
num_points = 1000
times = Time.now() + (u.Quantity(range(num_points)) * (2 * u.pi * (semi_major_axis**3 / 398600.4418 * u.km**3).to(u.s**2).value**0.5 / num_points) * u.s)

positions = []

for time in times:
    r = r.with_differentials(v)
    positions.append(r)

x_vals = [pos.x.value for pos in positions]
y_vals = [pos.y.value for pos in positions]
z_vals = [pos.z.value for pos in positions]

# Plot the orbit
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals)

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('LEO Orbit')

plt.show()