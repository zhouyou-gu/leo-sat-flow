import os

import matplotlib.pyplot as plt
import numpy as np

#create a figure with (2,1) subplots
FONT_SIZE = 9
fig_width_px = 350
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

ldl_path = "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_ld_starlink_varying_constellation_size/test_ld_starlink_varying_constellation_size-2025-October-23-21-39-03-ail/ldl"
sg_path = "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_ld_starlink_varying_constellation_size/test_ld_starlink_varying_constellation_size-2025-October-23-21-39-03-ail/sg"



# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

N_POINTS = 20

lml_data = np.genfromtxt(ldl_path, delimiter=',')
lml_res = -lml_data[:, 4].reshape(-1, N_POINTS)
lml_res = lml_res.mean(axis=1)


N_SG_STEP = 500
sg_data = np.genfromtxt(sg_path, delimiter=',')
sg_res = -sg_data[:, 4].reshape(-1, N_POINTS, N_SG_STEP)
sg_res = sg_res.mean(axis=1).reshape(-1, N_SG_STEP)
sg_res_selected = sg_res[:, [5-1, 10-1, 20-1, 50-1, 100-1, 200-1, 500-1]]

lml_res = lml_res[:sg_res_selected.shape[0]]

data = np.concatenate((
    lml_res.reshape(-1,1),
    sg_res_selected,
), axis=1)
data = data[0:5, :]

bar_width = 1/data.shape[0]
bars = []
index = np.arange(data.shape[0])*2

print(data)
# gradient colors start from the second color
# light purple to dark purple
colors = ["#0084ff", "#d7a7ff", "#b066ff", "#8a2bff", "#5c00ff", "#3a00cc", "#1a0099", "#000066", "#000033"]
data_name_list = [r"LaDuGL", r"ItSD-$5$", r"ItSD-$10$", r"ItSD-$20$", r"ItSD-$50$", r"ItSD-$100$", r"ItSD-$200$", r"ItSD-$500$"]
for i in range(data.shape[1]):
    b = axs.bar(index + (i+1.5)*bar_width-1, data[:, i], bar_width, label=data_name_list[i], zorder=3, color=colors[i])
    bars.append(b)
axs.set_position([0.175, 0.18, 0.8, 0.625])
axs.set_xticks(index)
print(index)
# axs.set_xticklabels(["500", "750", "1000", "1250", "1500", "1750", "2000", "2500", "3000"])
axs.set_xlabel(r'Number of Satellites, $I$')
axs.set_ylabel(r'Network Throughput (Gbps)')
axs.grid(True, zorder=0)
# axs.text(20, -5000, r'$\bf{(a)}$', fontsize=FONT_SIZE+2, verticalalignment='bottom')

# data_name_list = [r"SG $K=5$", r"SG $K=10$", r"SG $K=20$", r"SG $K=50$", r"SG $K=100$", r"MRate", r"LML-Trained GNN"]
ncol = 4  # Number of columns in the legend
h, l = bars, data_name_list
nrows = -(-len(h) // ncol)                         # ceiling division
idx   = np.arange(len(h))
pad   = nrows * ncol - len(idx)
idx   = np.concatenate([idx, np.full(pad, -1)])    # pad
order = idx.reshape(nrows, ncol).T.ravel()
order = order[order >= 0]                          # drop sentinels
h = [h[i] for i in order]
l = [l[i] for i in order]

leg = fig.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.175, 0.825, 0.8, 0.125), mode="expand",ncol = 4 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.2, 
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