import os

import matplotlib.pyplot as plt
import numpy as np

#create a figure with (2,1) subplots
FONT_SIZE = 9
fig_width_px = 400
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

results_dir = "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_ld_starlink_1000_constellation_varying/test_ld_starlink_1000_constellation_varying-2025-September-15-12-55-59-ail"

# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

N_POINTS = 100
res = np.zeros((len(results_dir), 4, N_POINTS))
result_list = []

sg_data = np.genfromtxt(os.path.join(results_dir, "sg"), delimiter=',')
ldl_data = np.genfromtxt(os.path.join(results_dir, "ldl"), delimiter=',')
heu_data = np.genfromtxt(os.path.join(results_dir, "heu"), delimiter=',')

AVG_N = 5
lines1 = []
avg = np.convolve(-sg_data[:,3], np.ones(AVG_N)/AVG_N, mode='full')[:N_POINTS]
print(avg)
line, = axs.plot(sg_data[:,2]/1e6, avg,linewidth=1, zorder=3)
lines1.append(line)

avg = np.convolve(-sg_data[:,4], np.ones(AVG_N)/AVG_N, mode='full')[:N_POINTS]
print(avg)
line, = axs.plot(sg_data[:,2]/1e6, avg,linewidth=1, zorder=3)
lines1.append(line)

line = axs.hlines(y=-ldl_data[3], xmin=1, xmax=N_POINTS, color='r', linestyle='--', linewidth=2)

lines1.append(line)

line = axs.hlines(y=-heu_data[3], xmin=1, xmax=N_POINTS, linestyle='-.', linewidth=2)

lines1.append(line)

# axs.set_position([0.18, 0.2, 0.3, 0.765])
axs.set_xlabel(r'Comptutation Time (s)')
axs.set_ylabel(r'Network Throughput (Gbps)')
# axs.set_xlim(0, N_POINTS)
# axs.set_ylim(-5000, -0)
axs.grid(True, zorder=0)
# axs.text(20, -5000, r'$\bf{(a)}$', fontsize=FONT_SIZE+2, verticalalignment='bottom')



data_name_list = [r"Subgradient",r"Subgradient- Ideal no time 0s", r"GNN - Dual Learning" + f"({ldl_data[2]/1e6:.2f} s)", r"MRate" + f"({heu_data[2]/1e6:.2f} s)"]
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

leg = axs.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.4, 0.025, 0.5, 0.3), mode="expand",ncol = 1 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.2, 
frameon=True,          # draw a frame
fancybox=False,        # <-- square corners (like MATLAB)
edgecolor='black',     # black  frame edge
facecolor='white',     # white background
framealpha=1,          # fully opaque
borderpad=0.3,         # tight inner padding (font-size units)
labelspacing=0.2,      # tight vertical space between rows
)   
leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border

# axs = axs_list[1]
# lines1 = []
# lines2 = []
# for i, b in enumerate(results_dir):
#     AVG_N = 5
#     avg = np.convolve(-res[i,1,:], np.ones(AVG_N)/AVG_N, mode='full')[:N_POINTS]
#     line, = axs.plot(np.arange(N_POINTS)+1, avg, linewidth=1., zorder=3)
#     lines1.append(line)
#     # line, = axs.plot(np.arange(N_POINTS)+1, res[i,1,:],linewidth=1)
#     # lines2.append(line)
#     # line, = axs.plot(np.arange(N_POINTS)+1, res[i,2,:],linewidth=1)
#     # lines.append(line)
#     # line, = axs.plot(np.arange(N_POINTS)+1, res[i,3,:],linewidth=1)
#     # lines.append(line)
# # line, = axs.plot(np.arange(N_POINTS)+1, -res[0,2,:],linewidth=1)
# # lines1.append(line)

# axs.set_position([0.63, 0.2, 0.33, 0.765])
# axs.set_xlabel(r'Number of Iterations, $K$')
# axs.set_ylabel(r"Network Throughput (Gbps)")
# axs.set_xlim(0, N_POINTS)
# axs.set_ylim(0, 180)
# axs.grid(True, zorder=0)
# axs.text(20, 0, r'$\bf{(b)}$', fontsize=FONT_SIZE+2, verticalalignment='bottom')


# data_name_list = [r"LDL", r"DPG", r"PG"]
# data_name_list.append(r"MRate")
# ncol = 1  # Number of columns in the legend
# h, l = lines1, data_name_list
# # nrows = -(-len(h) // ncol)                         # ceiling division
# # idx   = np.arange(len(h))
# # pad   = nrows * ncol - len(idx)
# # idx   = np.concatenate([idx, np.full(pad, -1)])    # pad
# # order = idx.reshape(nrows, ncol).T.ravel()
# # order = order[order >= 0]                          # drop sentinels
# # h = [h[i] for i in order]
# # l = [l[i] for i in order]

# leg = axs.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.4, 0.025, 0.5, 0.3), mode="expand",ncol = 1 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.2, 
# frameon=True,          # draw a frame
# fancybox=False,        # <-- square corners (like MATLAB)
# edgecolor='black',     # black  frame edge
# facecolor='white',     # white background
# framealpha=1,          # fully opaque
# borderpad=0.3,         # tight inner padding (font-size units)
# labelspacing=0.2,      # tight vertical space between rows
# )   
# leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border
# axs.add_artist(leg)

# Save the figure as a PDF
print("Saving figure to:", current_dir)
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'
fig.tight_layout(pad=0.1)
fig.savefig(output_path, format='pdf', pad_inches=0.)