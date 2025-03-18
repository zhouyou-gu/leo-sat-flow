import numpy as np
import matplotlib.pyplot as plt

# Instead of using pyrate.propagation, try importing the propagation function directly.
# Replace 'pyrate.propagator' with the correct submodule if needed.
from pyrateoptics import fresnel_propagate

# ----------------------------
# Simulation parameters
# ----------------------------
L = 10e-3       # total grid size (10 mm)
N = 256         # grid points per dimension
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# Gaussian beam parameters
w0 = 1e-3       # beam waist (1 mm)
field_initial = np.exp(- (X**2 + Y**2) / w0**2)

# Propagation parameters
z = 1.0               # propagation distance in meters
wavelength = 1550e-9  # wavelength in meters

# ----------------------------
# Propagate the beam in free space
# ----------------------------
# Use the imported fresnel_propagate function
field_propagated = fresnel_propagate(field_initial, z, L, wavelength)

# ----------------------------
# Introduce lateral (tracking) error
# ----------------------------
dx_error = 1e-3  # shift by 1 mm along x-axis
shift_pixels = int(dx_error / (L / N))
field_shifted = np.roll(field_propagated, shift=shift_pixels, axis=1)

# ----------------------------
# Compute intensity distributions
# ----------------------------
intensity_no_error = np.abs(field_propagated)**2
intensity_with_error = np.abs(field_shifted)**2

# ----------------------------
# Compute coupling efficiency into a circular receiver aperture
# ----------------------------
aperture_radius = 1e-3  # 1 mm radius
aperture_mask = (X**2 + Y**2) <= aperture_radius**2

def coupling_efficiency(intensity, mask):
    total_power = np.sum(intensity)
    power_in_aperture = np.sum(intensity[mask])
    return power_in_aperture / total_power

eff_no_error = coupling_efficiency(intensity_no_error, aperture_mask)
eff_with_error = coupling_efficiency(intensity_with_error, aperture_mask)

# ----------------------------
# Visualization
# ----------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
im0 = ax[0].imshow(intensity_no_error, extent=[x[0], x[-1], y[0], y[-1]])
ax[0].set_title(f"Without Tracking Error\nCoupling Efficiency: {eff_no_error:.3f}")
ax[0].set_xlabel('x (m)')
ax[0].set_ylabel('y (m)')
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

im1 = ax[1].imshow(intensity_with_error, extent=[x[0], x[-1], y[0], y[-1]])
ax[1].set_title(f"With Tracking Error\nCoupling Efficiency: {eff_with_error:.3f}")
ax[1].set_xlabel('x (m)')
ax[1].set_ylabel('y (m)')
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
