import os

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

#create a figure with (2,1) subplots
FONT_SIZE = 9
fig_width_px = 400
fig_height_px = 210
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
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def force_square(ax, equal_units=True):
    # 1) Make the axes box square
    ax.set_box_aspect(1)  # matplotlib ≥ 3.3

    if equal_units:
        # 2) Make data units equal and pad limits so ranges match
        ax.set_aspect('equal', adjustable='box')
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xr, yr = x1 - x0, y1 - y0
        r = max(xr, yr)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(cx - r / 2, cx + r / 2)
        ax.set_ylim(cy - r / 2, cy + r / 2)

current_dir = os.path.dirname(os.path.abspath(__file__))

data_file = os.environ.get(
    "LEO_SAT_FLOW_FIG8_DATA",
    "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_lemma_1_rand_weights/test_lemma_1_rand_weights-2025-September-15-11-09-58-ail/d_o_component",
)
# Plot the data
fig, axs_list = plt.subplots(1,2,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

N_POINTS = 200
data = np.genfromtxt(data_file, delimiter=',')
print(data)

axs = axs_list[0]
axs.set_position([0.115, 0.21, 0.34, 0.62])

axs.scatter(-data[:,3], -data[:,5], s=1)
axs.set_title('(i) Matching term', pad=2)
axs.set_xlabel(r'Before clipping ($\times 10^4$)',
                labelpad=5,       # distance from axis
                loc='center')      # align: 'center', 'top', or 'bottom'
axs.set_ylabel(r'After clipping ($\times 10^4$)',
              labelpad=5,       # distance from axis
              loc='center')      # align: 'center', 'top', or 'bottom'
axs.set_xlim(2000,10000)
axs.set_ylim(2000,10000)
axs.set_xticks([2000, 4000, 6000, 8000, 10000])
axs.set_yticks([2000, 4000, 6000, 8000, 10000])
axs.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
axs.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
axs.plot([2000, 10000], [2000, 10000], color='red', linestyle='--', linewidth=0.5)

axs.grid(True, zorder=0)
axs.set_aspect('equal', 'box')
#plot a diagonal line

force_square(axs, equal_units=True)

axs = axs_list[1]
axs.set_position([0.62, 0.21, 0.34, 0.62])

axs.scatter(data[:,2], data[:,4], s=1)
axs.set_title('(ii) Routing and rate term', pad=2)
axs.set_xlabel('Before clipping',
                labelpad=5,       # distance from axis
                loc='center')      # align: 'center', 'top', or 'bottom'
axs.set_ylabel('After clipping',
              labelpad=5,       # distance from axis
              loc='center')      # align: 'center', 'top', or 'bottom'
axs.set_xlim(-125,-50)
axs.set_xticks([-125, -100, -75, -50])
axs.set_ylim(-125,-50)  
axs.set_yticks([-125, -100, -75, -50])
axs.plot([-125, -50], [-125, -50], color='red', linestyle ='--', linewidth=0.5)

axs.grid(True, zorder=0)
axs.set_aspect('equal', 'box')

#plot a diagonal line

force_square(axs, equal_units=True)

# Save the figure as a PDF
print("Saving figure to:", current_dir)
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)
