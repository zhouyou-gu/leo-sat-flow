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

# ldl_path = "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_ld_starlink_varying_constellation_size/test_ld_starlink_varying_constellation_size-2025-October-23-21-39-03-ail/ldl"

baseline_ldl_path = os.environ.get(
    "CONSTELLATION_BASELINE_LDL",
    "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_ld_starlink_1000_varying_constellation/test_ld_starlink_1000_varying_constellation-2026-January-05-11-24-31-ail/ldl",
)
kuiper_ldl_path = os.environ.get(
    "CONSTELLATION_KUIPER_LDL",
    "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/test_ld_starlink_1000_varying_constellation/test_ld_starlink_1000_varying_constellation-2026-July-24-12-58-18-ail/ldl",
)


# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

N_POINTS = 2

def extract_method_averages(path):
    ldl_data = np.genfromtxt(path, delimiter=',')
    if ldl_data.ndim != 2 or ldl_data.shape[0] % N_POINTS:
        raise ValueError(f"Unexpected result shape {ldl_data.shape} in {path}")
    if not np.isfinite(ldl_data).all():
        raise ValueError(f"Non-finite result in {path}")
    method_columns = [4, 7, 8, 6, 9]
    return -ldl_data[:, method_columns].reshape(-1, N_POINTS, len(method_columns)).mean(axis=1)


baseline_data = extract_method_averages(baseline_ldl_path)
kuiper_data = extract_method_averages(kuiper_ldl_path)
if baseline_data.shape[0] < 3 or kuiper_data.shape[0] < 1:
    raise ValueError("The result files do not contain the required constellation rows")

# Preserve the published Starlink and OneWeb cases and append the new Kuiper case.
data = np.vstack((baseline_data[0], baseline_data[2], kuiper_data[-1]))
bar_width = 0.275
bars = []

print(data)

data_name_list = [r"DeepLaDu", r"Random", r"+Grid", r"MRate", r"SaTE"]
index = np.arange(data.shape[0])*2

for i in range(data.shape[1]):
    b = axs.bar(index + (i -2)* bar_width, data[:, i], bar_width, label=data_name_list[i], zorder=3)
    bars.append(b)
axs.set_position([0.175, 0.18, 0.8, 0.7])
axs.set_xticks(index)
print(index)
axs.set_xticklabels(["Starlink", "OneWeb", "Kuiper"])
axs.set_xlabel(r'Constellation Type')
axs.set_ylabel(r'Network Throughput (Gbps)')
axs.grid(True, zorder=0)
# axs.text(20, -5000, r'$\bf{(a)}$', fontsize=FONT_SIZE+2, verticalalignment='bottom')

# data_name_list = [r"SG $K=5$", r"SG $K=10$", r"SG $K=20$", r"SG $K=50$", r"SG $K=100$", r"MRate", r"LML-Trained GNN"]
ncol = 1  # Number of columns in the legend
h, l = bars, data_name_list
# nrows = -(-len(h) // ncol)                         # ceiling division
# idx   = np.arange(len(h))
# pad   = nrows * ncol - len(idx)
# idx   = np.concatenate([idx, np.full(pad, -1)])    # pad
# order = idx.reshape(nrows, ncol).T.ravel()
# order = order[order >= 0]                          # drop sentinels
# h = [h[i] for i in order]
# l = [l[i] for i in order]

leg = fig.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.175, 0.9, 0.8, 0.125), mode="expand",ncol = 5 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.2, 
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
