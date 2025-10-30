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

LML_PATH = "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_ld_starlink_1000_varying_jitter/test_ld_starlink_1000_varying_jitter-2025-October-29-14-53-41-ail/ldl"


# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect 

lml_data = np.genfromtxt(LML_PATH, delimiter=',')
lml_res = -lml_data[:, 6].reshape(-1,10)

markers = ['o', 's', '^', '+', 'x', '>', 'p', '*', 'h']
lines1 = []
for i in range(lml_res.shape[0]):
    if i == 0:
        continue
    l, = axs.plot((np.arange(lml_res.shape[1])+1)*50, lml_res[i], marker=markers[i], markersize=5, zorder=5)
    lines1.append(l)

axs.set_position([0.15, 0.175, 0.825, 0.65])
axs.set_ylabel(r'Network Throughput (Gbps)')
axs.set_xlabel(r'Beam Angular Spread ($\mu$rad)')
axs.set_xlim(50, 500)
# axs.set_ylim(0, 1)
axs.grid(True, zorder=0)
# axs.text(20, -5000, r'$\bf{(a)}$', fontsize=FONT_SIZE+2, verticalalignment='bottom')

data_name_list = [r"$\sigma_\mathrm{J}=10\ \mu$rad", r"$\sigma_\mathrm{J}=20\ \mu$rad", r"$\sigma_\mathrm{J}=30\ \mu$rad", r"$\sigma_\mathrm{J}=40\ \mu$rad", r"$\sigma_\mathrm{J}=50\ \mu$rad"]
ncol = 3  # Number of columns in the legend
h, l = lines1, data_name_list
nrows = -(-len(h) // ncol)                         # ceiling division
idx   = np.arange(len(h))
pad   = nrows * ncol - len(idx)
idx   = np.concatenate([idx, np.full(pad, -1)])    # pad
order = idx.reshape(nrows, ncol).T.ravel()
order = order[order >= 0]                          # drop sentinels
h = [h[i] for i in order]
l = [l[i] for i in order]

leg = fig.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.15, 0.845, 0.825, 0.175), mode="expand",ncol = ncol ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.2, 
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