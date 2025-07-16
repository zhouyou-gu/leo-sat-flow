import os
from sim_mld.solver import mr_solver
from sim_mld.lisl_channel_model import *

import matplotlib.pyplot as plt
import numpy as np

from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT


z_values = np.geomspace(1, 3e6, 100)  # from 0 to 3000 km


# solver parameters
wavelength = mr_solver.WAVELENGTH  # Wavelength in meters (1.55 microns typical in telecom)
w0 = w0_from_angular_spreading(mr_solver.ANGULAR_SPREADING, wavelength)  # Beam waist in meters
P = mr_solver.PEAK_POWER_W
B = mr_solver.BANDWIDTH  # 1 GHz bandwidth
R = mr_solver.RESPONSIVITY     # A/W responsivity
A = mr_solver.APERTURE_AREA  # Area in m^2 (example)
N0 = mr_solver.NOISE_CURRENT  # Example noise in A rms

# Compute intensity vs. z at rho=0
I_on_axis = np.array([gaussian_beam_intensity(P, w0, wavelength, 0, z) for z in z_values])

# Compute capacity lower bound vs. z at rho=0
C_on_axis = np.array([capacity_lower_bound(B, R, A, N0, P, w0, wavelength, 0, z) for z in z_values])

sigma_jitter = mr_solver.JITTER  # Jitter in radians
epsilon = mr_solver.EPSILON
I_relaxed = intensity_relaxed(P, w0, wavelength, z_values, sigma_jitter, epsilon)
C_relaxed = capacity_relaxed(B, R, A, N0, P, w0, wavelength, z_values, sigma_jitter, epsilon)

print("Intensity on Axis:", I_on_axis)
print("Capacity Lower Bound on Axis:", C_on_axis)
print("Intensity with Jitter:", I_relaxed)
print("Capacity with Jitter:", C_relaxed)

#create a figure with (2,1) subplots
FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 250
dpi = 100  # Typical screen DPI, adjust if necessary
fig_width_in = fig_width_px / dpi
fig_height_in = fig_height_px / dpi

plt.rcParams.update({
    # --- make every normal string use Times or the best fall-back ----
    'font.family' : 'serif',
    'font.serif'  : ['Times New Roman', 'Times', 'Nimbus Roman', 'STIXGeneral'],
    # --- keep your existing sizing rules ----------------------------
    'font.size'        : FONT_SIZE,
    'axes.titlesize'   : FONT_SIZE,
    'axes.labelsize'   : FONT_SIZE,
    'xtick.labelsize'  : FONT_SIZE,
    'ytick.labelsize'  : FONT_SIZE,
    'legend.fontsize'  : FONT_SIZE,
})
plt.rc('mathtext', fontset='cm')

current_dir = os.path.dirname(os.path.abspath(__file__))

# Plot the data
fig, axs = plt.subplots(1,2,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

# Plot settings
markers = ['o', 's', '^','+']  # Different markers for each line
linesty = [':', '-',(0, (3, 1, 1, 1)), (0, (5, 1)),'-','-','-']  # Different markers for each line
p_names = [r'a',r'b']
colors = ["#F8BF91","#FF9137","#FF7300","#C75A00","#723300"]

cell_size_list = [10]
bars = []

lines = []
l, = axs[0].plot(z_values/1e3, I_on_axis)
lines.append(l)
l, = axs[0].plot(z_values/1e3, I_relaxed)
lines.append(l)
axs[0].set_title('Beam Intensity')
axs[0].set_xlabel('Distance (Km)')
axs[0].set_ylabel(r'Intensity (W/m$^2$)')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].grid()
axs[0].set_position([0.16, 0.2, 0.34, 0.7])
axs[0].set_xlim(1, 3e3)
axs[0].set_xticks([0.01, 1, 100, 10000])
# Add a legend
leg = axs[0].legend(lines, ["On Axis", "With Jitter"] ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.04, 0.075, 0.65, 0.3), mode="expand",ncol = 1 ,borderaxespad=0.,handlelength=1, handleheight= 1, handletextpad=0.2, 
frameon=True,          # draw a frame
fancybox=False,        # <-- square corners (like MATLAB)
edgecolor='black',     # black  frame edge
facecolor='white',     # white background
framealpha=1,          # fully opaque
borderpad=0.3,         # tight inner padding (font-size units)
labelspacing=0.2,      # tight vertical space between rows
)   
leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border


lines = []
l, = axs[1].plot(z_values/1e3, C_on_axis)
lines.append(l)
l, = axs[1].plot(z_values/1e3, C_relaxed)
lines.append(l)
axs[1].set_title('Link Rate')
axs[1].set_xlabel('Distance (Km)')
axs[1].set_ylabel('Rate (Gbits/s)')
axs[1].set_xscale('log')
axs[1].grid()
axs[1].set_position([0.6225, 0.2, 0.34, 0.7])
axs[1].set_xlim(1, 3e3)
axs[1].set_xticks([0.01, 1, 100, 10000])
# Add a legend
leg = axs[1].legend(lines, ["On Axis", "With Jitter"] ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.04, 0.075, 0.65, 0.3), mode="expand",ncol = 1 ,borderaxespad=0.,handlelength=1, handleheight= 1, handletextpad=0.2, 
frameon=True,          # draw a frame
fancybox=False,        # <-- square corners (like MATLAB)
edgecolor='black',     # black  frame edge
facecolor='white',     # white background
framealpha=1,          # fully opaque
borderpad=0.3,         # tight inner padding (font-size units)
labelspacing=0.2,      # tight vertical space between rows
)   
leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border

# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)