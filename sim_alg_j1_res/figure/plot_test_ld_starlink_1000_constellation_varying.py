import os

import matplotlib.pyplot as plt
import numpy as np

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
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

current_dir = os.path.dirname(os.path.abspath(__file__))

results_dir_name = "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_ld_starlink_1000_constellation_varying/test_ld_starlink_1000_constellation_varying-2025-September-15-21-25-28-ail"

SG_PATH = os.path.join(results_dir_name, "sg")
HEU_PATH = os.path.join(results_dir_name, "heu")
LML_PATH = os.path.join(results_dir_name, "ldl")

results_dir = [SG_PATH, HEU_PATH, LML_PATH]

# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

N_POINTS = 100
res = np.zeros((len(results_dir), 2, N_POINTS))

sg_data = np.genfromtxt(SG_PATH, delimiter=',')
sg_res = -sg_data[:, 3]

sg_res_100 = sg_res[(5-1)::500]
sg_res_200 = sg_res[(10-1)::500]
sg_res_300 = sg_res[(20-1)::500]
sg_res_400 = sg_res[(50-1)::500]
sg_res_500 = sg_res[(100-1)::500]

lml_data = np.genfromtxt(LML_PATH, delimiter=',')
lml_res = -lml_data[:, 3]

heu_data = np.genfromtxt(HEU_PATH, delimiter=',')
heu_res = -heu_data[:, 3]

MARKER_SIZE = 4
#plot cdf of the data, with sparse markers
lines1 = []
l, = axs.plot(np.sort(sg_res_100), np.linspace(0, 1, len(sg_res_100), endpoint=False), marker='o', markersize=MARKER_SIZE, linewidth=1, markevery=10)
lines1.append(l)
l, = axs.plot(np.sort(sg_res_200), np.linspace(0, 1, len(sg_res_200), endpoint=False), marker='s', markersize=MARKER_SIZE, linewidth=1, markevery=10)
lines1.append(l)
l, = axs.plot(np.sort(sg_res_300), np.linspace(0, 1, len(sg_res_300), endpoint=False), marker='^', markersize=MARKER_SIZE, linewidth=1, markevery=10)
lines1.append(l)
l, = axs.plot(np.sort(sg_res_400), np.linspace(0, 1, len(sg_res_400), endpoint=False), marker='x', markersize=MARKER_SIZE, linewidth=1, markevery=10)
lines1.append(l)
l, = axs.plot(np.sort(sg_res_500), np.linspace(0, 1, len(sg_res_500), endpoint=False), marker='*', markersize=MARKER_SIZE, linewidth=1, markevery=10)
lines1.append(l)
l, = axs.plot(np.sort(heu_res), np.linspace(0, 1, len(heu_res), endpoint=False), marker='D', markersize=MARKER_SIZE, linewidth=1, markevery=10)
lines1.append(l)
l, = axs.plot(np.sort(lml_res), np.linspace(0, 1, len(lml_res), endpoint=False), marker='v', markersize=MARKER_SIZE, linewidth=1, markevery=10)
lines1.append(l)
axs.set_position([0.15, 0.175, 0.8, 0.75])
axs.set_xlabel(r'Network Throughput (Gbps)')
axs.set_ylabel(r'CDF')
axs.set_xlim(50, 160)
axs.set_ylim(0, 1)
axs.grid(True, zorder=0)
# axs.text(20, -5000, r'$\bf{(a)}$', fontsize=FONT_SIZE+2, verticalalignment='bottom')

data_name_list = [r"SG $K=5$", r"SG $K=10$", r"SG $K=20$", r"SG $K=50$", r"SG $K=100$", r"MRate", r"LML-Trained GNN"]
ncol = 1  # Number of columns in the legend
h, l = lines1, data_name_list
# nrows = -(-len(h) // ncol)                         # ceiling division
# idx   = np.arange(len(h))
# pad   = nrows * ncol - len(idx)
# idx   = np.concatenate([idx, np.full(pad, -1)])    # pad
# order = idx.reshape(nrows, ncol).T.ravel()
# order = order[order >= 0]                          # drop sentinels
# h = [h[i] for i in order]
# l = [l[i] for i in order]

leg = axs.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.55, 0.025, 0.435, 0.3), mode="expand",ncol = 1 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.2, 
frameon=True,          # draw a frame
fancybox=False,        # <-- square corners (like MATLAB)
edgecolor='black',     # black  frame edge
facecolor='white',     # white background
framealpha=1,          # fully opaque
borderpad=0.3,         # tight inner padding (font-size units)
labelspacing=0.2,      # tight vertical space between rows
)   
leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border
axs.add_artist(leg)

# Save the figure as a PDF
print("Saving figure to:", current_dir)
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)