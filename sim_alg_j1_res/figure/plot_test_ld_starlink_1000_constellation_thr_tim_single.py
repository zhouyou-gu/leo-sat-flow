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

results_dir_name = "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_ld_starlink_1000_constellation_varying/test_ld_starlink_1000_constellation_varying-2025-October-27-12-56-15-ail"

SG_PATH = os.path.join(results_dir_name, "sg")
HEU_PATH = os.path.join(results_dir_name, "heu")
LML_PATH = os.path.join(results_dir_name, "ldl")

results_dir = [SG_PATH, HEU_PATH, LML_PATH]

# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

LINE_WIDTH = 1.5
# regular_matplot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
colors = ['#1f77b4', '#d62728',"#D6B7FF", "#B378FF", "#934BFF", "#8E40FB", "#6A00FF"]
markers = ['d', 's', '^', 'p', 'v', '>', 'p', '*', 'h']
markers_n_sat = ['+',  '1', '2', '3', '4']
lines2 = []
marker_size = 50
for i, N_SAT in enumerate([500, 750, 1000, 1250, 1500]):
    sg_data = np.genfromtxt(SG_PATH, delimiter=',')[i*500*10:(i+1)*500*10, :]
    sg_res = -sg_data[:, 3].reshape(-1,500).mean(axis=0)
    sg_time = sg_data[:, 2].reshape(-1,500).mean(axis=0)/1e6
    sg_res = sg_res.squeeze()
    sg_time = sg_time.squeeze()
    sg_res = sg_res[np.array([20, 100, 300])-1]
    sg_time = sg_time[np.array([20, 100, 300])-1]

    lml_data = np.genfromtxt(LML_PATH, delimiter=',')[i*10:(i+1)*10, :]
    lml_res = -lml_data[:, 3].reshape(-1,10).mean(axis=1)
    lml_time = lml_data[:, 2].reshape(-1,10).mean(axis=1)/1e6

    heu_data = np.genfromtxt(HEU_PATH, delimiter=',')[i*10:(i+1)*10, :]
    heu_res = -heu_data[:, 3].reshape(-1,10).mean(axis=1)
    heu_time = heu_data[:, 2].reshape(-1,10).mean(axis=1)/1e6
    
    lines1 = []
    lines3 = []
    counter = 0
    for j in range(lml_res.shape[0]):
        l = axs.scatter(lml_time[j], lml_res[j], marker=markers[i], s=marker_size, zorder=5, color=colors[counter])
        lines1.append(l)
        l = axs.scatter(1e6, 1e6, marker='o', s=marker_size, zorder=0,color=colors[counter])
        lines3.append(l)
        # l = axs.scatter(lml_time[j], lml_res[j], marker=markers[i], s=marker_size, zorder=0, color='black')
        counter += 1
    for j in range(heu_res.shape[0]):
        l = axs.scatter(heu_time[j], heu_res[j], marker=markers[i], s=marker_size, zorder=5, color=colors[counter])
        lines1.append(l)
        l = axs.scatter(1e6, 1e6, marker='o', s=marker_size, zorder=0,color=colors[counter])
        lines3.append(l)
        # l = axs.scatter(heu_time[j], heu_res[j], marker=markers[i], s=marker_size, zorder=0, color='black')
        counter += 1
    for j in range(sg_res.shape[0]):
        l = axs.scatter(sg_time[j], sg_res[j], marker=markers[i], s=marker_size, zorder=5, color=colors[counter])
        lines1.append(l)
        l = axs.scatter(1e6, 1e6, marker='o', s=marker_size, zorder=0,color=colors[counter])
        lines3.append(l)

        # l = axs.scatter(sg_time[j], sg_res[j], marker=markers[i], s=marker_size, zorder=0, color='black')
        counter += 1
    l = axs.scatter(1e6, 1e6, marker=markers[i], s=marker_size, zorder=0, facecolors='none', edgecolors='black' )
    lines2.append(l)

axs.set_position([0.125, 0.175, 0.825, 0.65])
axs.set_xlabel(r'Computation Time (s)')
axs.set_ylabel(r'Network Throughput (Gbps)')
axs.set_xlim(0.001, 10000)
axs.set_xscale("log")
axs.set_ylim(30, 180)
axs.grid(True, zorder=0)
# add a vertical line at x = 1; with a text "1 second" on the side of the line in vertical alignment bottom
axs.axvline(x=1, color='black', linestyle='-', linewidth=2.5, zorder=0)
axs.text(1.2, 35, r'     Coherent time $\approx$ 1 second ', rotation=90, verticalalignment='bottom', color='black', fontsize=FONT_SIZE)

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
data_name_list  = [r'DeepLaDu', r'MRate', r'LaDu-20', r'LaDu-100', r'LaDu-200', r'$I=500$', r'$I=750$', r'$I=1000$', r'$I=1250$', r'$I=1500$']
ncol = 5  # Number of columns in the legend

h, l = lines3, data_name_list
h.extend(lines2)
nrows = -(-len(h) // ncol)                         # ceiling division
idx   = np.arange(len(h))
pad   = nrows * ncol - len(idx)
idx   = np.concatenate([idx, np.full(pad, -1)])    # pad
order = idx.reshape(nrows, ncol).T.ravel()
order = order[order >= 0]                          # drop sentinels
h = [h[i] for i in order]
l = [l[i] for i in order]

leg = fig.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.125, 0.835, 0.825, 0.1), mode="expand",ncol = ncol ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.15, 
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