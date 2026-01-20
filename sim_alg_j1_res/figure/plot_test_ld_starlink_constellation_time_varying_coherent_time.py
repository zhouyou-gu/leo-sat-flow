import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np

#create a figure with (2,1) subplots
FONT_SIZE = 9
fig_width_px = 400
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
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

current_dir = os.path.dirname(os.path.abspath(__file__))

results_dir_name = "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_ld_starlink_constellation_time_varying_coherent_time/test_ld_starlink_constellation_time_varying_coherent_time-2025-November-25-10-32-17-ail/rm_links"



# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

LINE_WIDTH = 0.5
colors = ["#D6B7FF", "#B378FF", "#934BFF", "#8E40FB", "#6A00FF"]
markers = ['x', '+', 'o', 'd', 's', '>', 'p', '*', 'h']
markers_n_sat = ['+',  '1', '2', '3', '4']
lines2 = []
marker_size = 50

data = np.genfromtxt(results_dir_name, delimiter=',')
total_links = data[:,5]
num_removed = data[:,4]
for i, N_SAT in enumerate([500, 750, 1000, 1250, 1500]):
    sat_data = data[data[:,2]==N_SAT]
    sat_data_total_links = sat_data[:,5].reshape(-1,16).mean(axis=0)
    sat_data_num_removed = sat_data[:,4].reshape(-1,16).mean(axis=0)
    TIME = [1e3,2e3,5e3,1e4,2e4,5e4,1e5,2e5,5e5,1e6,2e6,5e6,1e7,2e7,5e7,1e8]
    TIME = np.array(TIME)
    TIME = TIME / 1e6  # convert to milliseconds
    sat_data_total_links = sat_data_total_links.squeeze()
    sat_data_num_removed = sat_data_num_removed.squeeze()
    ratio_removed = sat_data_num_removed / sat_data_total_links
    print(ratio_removed)
    # no fill in markers
    l, = axs.plot(TIME, ratio_removed*100, marker=markers[i], markersize=5, linewidth=LINE_WIDTH, zorder=3, markerfacecolor='none')
    lines2.append(l)
    

axs.set_position([0.125, 0.175, 0.85, 0.7])
axs.set_xlabel(r'Time Elapsed (s)')
axs.set_ylabel(r'Percentage of Lost Connectable Links (%)')
# axs.set_xlim(0.001, 10000)
axs.set_xscale("log")
# axs.set_ylim(30, 180)
axs.grid(True, zorder=0)
axs.axvline(x=1, color='black', linestyle='-', linewidth=2.5, zorder=0)
axs.text(1.2, 0, r'     Coherent time $\approx$ 1 second ', rotation=90, verticalalignment='bottom', color='black', fontsize=FONT_SIZE)

# x_tail = 0.2
# y_tail = 0.1
# x_head = 0.95
# y_head = 0.95
# dx = x_head - x_tail
# dy = y_head - y_tail
# arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (dx, dy),
#                                  mutation_scale=100,
#                                  transform=axs.transAxes, zorder=5)
# arrows = axs.add_patch(arrow)
data_name_list  = [ r'$I=500$', r'$I=750$', r'$I=1000$', r'$I=1250$', r'$I=1500$']
ncol = 5  # Number of columns in the legend

h, l = lines2, data_name_list
nrows = -(-len(h) // ncol)                         # ceiling division
idx   = np.arange(len(h))
pad   = nrows * ncol - len(idx)
idx   = np.concatenate([idx, np.full(pad, -1)])    # pad
order = idx.reshape(nrows, ncol).T.ravel()
order = order[order >= 0]                          # drop sentinels
h = [h[i] for i in order]
l = [l[i] for i in order]

leg = fig.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.125, 0.895, 0.85, 0.1), mode="expand",ncol = ncol ,borderaxespad=0.,handlelength=1.25, handleheight= 0.8, handletextpad=0.35, 
frameon=True,          # draw a frame
fancybox=False,        # <-- square corners (like MATLAB)
edgecolor='black',     # black  frame edge
facecolor='white',     # white background
framealpha=1,          # fully opaque
borderpad=0.3,         # tight inner padding (font-size units)
labelspacing=0.2,      # tight vertical space between rows
)   
leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border
fig.add_artist(leg)

# Save the figure as a PDF
print("Saving figure to:", current_dir)
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)