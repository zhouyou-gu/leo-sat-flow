import os

import matplotlib.pyplot as plt
import numpy as np

#create a figure with (2,1) subplots
FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 200
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
results_file = "/home/zhouyou/leo-sat-flow/sim_alg_v1_res/test_dual_optimization_different_constellation/test_dual_optimization_different_constellation-2025-July-25-21-52-32-ail/res.csv"

# Plot the data
fig, axs = plt.subplots(1,1)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

name = ["DuJo","+Grid","Rand","MRate"]
data = np.genfromtxt(results_file, delimiter=',')
markers = ['o', 's', 'D', '^', 'v', 'x', '*']

data = -data[:, [4,5,7,6]]  # Select the relevant columns and negate them
data = data[[1,2,0],:]
bar_width = 0.4
bars = []
index = np.arange(data.shape[0])*2
for i in range(data.shape[1]):
    b = axs.bar(index + (i -1.5)* bar_width, data[:, i], bar_width, label=name[i], zorder=3)
    bars.append(b)
axs.set_position([0.175, 0.125, 0.8, 0.725])
axs.set_ylabel(r"Network Throughput (Gbps)")
axs.set_xticks(index)
axs.set_xticklabels([r"Starlink", r"Walker-Delta", r"OneWeb"])
axs.grid(True, zorder=0)
# axs.set_position([0.18, 0.2, 0.775, 0.765])
# # axs.set_title('Beam Intensity')
# axs.set_xlabel('Number of Iterations')
# axs.set_ylabel(r'$g(\lambda)$')
# axs.set_xlim(1, N_POINTS)
# axs.set_ylim(-1600, -400)
# # axs.set_xscale('log')
# # axs.set_yscale('log')
# axs.grid()
# # Add a legend

# data_name_list = [r"$\beta=$"+f"{0.1+0.2*i:.1f}" for i in range(5)] 
# ncol = 3  # Number of columns in the legend
# h, l = lines1, data_name_list
# nrows = -(-len(h) // ncol)                         # ceiling division
# idx   = np.arange(len(h))
# pad   = nrows * ncol - len(idx)
# idx   = np.concatenate([idx, np.full(pad, -1)])    # pad
# order = idx.reshape(nrows, ncol).T.ravel()
# order = order[order >= 0]                          # drop sentinels
# h = [h[i] for i in order]
# l = [l[i] for i in order]

leg = fig.legend(bars, name ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.175, 0.875, 0.8, 0.125), mode="expand",ncol = 4 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.2, 
frameon=True,          # draw a frame
fancybox=False,        # <-- square corners (like MATLAB)
edgecolor='black',     # black  frame edge
facecolor='white',     # white background
framealpha=1,          # fully opaque
borderpad=0.3,         # tight inner padding (font-size units)
labelspacing=0.2,      # tight vertical space between rows
)   
leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border
# fig.add_artist(leg)

# leg = plt.legend(lines2, [r"$\beta=$"+f"{0.1+0.2*i:.1f}" for i in range(5)] ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.125, 0.8, 0.7, 0.3), mode="expand",ncol = 3 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.2, 
# frameon=True,          # draw a frame
# fancybox=False,        # <-- square corners (like MATLAB)
# edgecolor='black',     # black  frame edge
# facecolor='white',     # white background
# framealpha=1,          # fully opaque
# borderpad=0.5,         # tight inner padding (font-size units)
# labelspacing=0.1,      # tight vertical space between rows
# )   
# leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border
# axs.add_artist(leg)

# Save the figure as a PDF
print("Saving figure to:", current_dir)
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)