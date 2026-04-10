import os

import matplotlib.pyplot as plt
import numpy as np

#create a figure with (2,1) subplots
FONT_SIZE = 9
fig_width_px = 400
fig_height_px = 225
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

# Plot the data
fig, axs_list = plt.subplots(1,2)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio



results_file = "/home/zhouyou/leo-sat-flow/sim_alg_v1_res/test_dual_optimization_starlink_1000_varying_for/test_dual_optimization_starlink_1000_varying_for-2025-July-31-20-51-49-ail/res.csv"
name = ["DuJo","+Grid","Rand","MRate"]
data = np.genfromtxt(results_file, delimiter=',')
markers = ['o', 's', 'D', '^', 'v', 'x', '*']
axs = axs_list[0]
data = -data[:6, [4,5,7,6]]  # DuJo, +Grid, Rand, MRate
bar_width = 0.4
bars = []
index = np.arange(data.shape[0])*2
for i in range(data.shape[1]):
    b = axs.bar(index + (i -1.5)* bar_width, data[:, i], bar_width, label=name[i], zorder=3)
    bars.append(b)
axs.set_position([0.575, 0.18, 0.415, 0.7])
axs.set_yticks([0, 50, 100, 150, 200])
axs.set_yticklabels([])
axs.set_xlabel(r"Field of Regard Size (Degree)")
axs.set_xticks(index)
axs.set_xticklabels(["30", "40", "50", "60", "70", "80"])
axs.grid(True, zorder=0)


results_file = "/home/zhouyou/leo-sat-flow/sim_alg_v1_res/test_dual_optimization_mixed_constellation_lct_failure_rate/test_dual_optimization_mixed_constellation_lct_failure_rate-2025-July-23-20-18-16-ail/res.csv"
name = ["DuJo","+Grid","Rand","MRate"]
data = np.genfromtxt(results_file, delimiter=',')
markers = ['o', 's', 'D', '^', 'v', 'x', '*']
axs = axs_list[1]
data = -data[:, [4,5,7,6]]  # DuJo, +Grid, Rand, MRate
bar_width = 0.4
bars = []
index = np.arange(data.shape[0])*2
for i in range(data.shape[1]):
    b = axs.bar(index + (i -1.5)* bar_width, data[::-1, i], bar_width, label=name[i], zorder=3)
    bars.append(b)
axs.set_position([0.125, 0.18, 0.415, 0.7])
# axs.set_ylabel(r"Network Throughput (Gbps)")
axs.set_yticks([0, 50, 100, 150, 200])
axs.set_yticklabels([0, 50, 100, 150, 200])
axs.set_ylabel(r"Network Throughput (Gbps)")
axs.set_xlabel(r"Average Number of LCTs per Sat.")
axs.set_xticks(index)
axs.set_xticklabels(["0.8", "1.0", "1.2", "1.4", "1.6", "1.8", "2.0"])
axs.grid(True, zorder=0)

leg = fig.legend(bars, name ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.125, 0.9, 0.865, 0.125), mode="expand",ncol = 4 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.2, 
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
