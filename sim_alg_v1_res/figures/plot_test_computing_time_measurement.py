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


current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, os.path.pardir, "test_computing_time_measurement")

# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

n_sat = [500, 750, 1000, 1250, 1500, 1750, 2000]
N_POINTS = 20
res = np.zeros((len(n_sat), 7, N_POINTS))
result_list = []
for i, n in enumerate(n_sat):    
    print(f"Processing n_sat = {n}")
    tmp_dir = "test_computing_time_measurement-2025-July-23-15-49-08-ail"
    fname = f"lpdsolver.dual_matching_time.n_sat{n}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    res[i, 0, :] = data[:, 3]
    
    fname = f"lpdsolver.dual_srouting_time.n_sat{n}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    res[i, 1, :] = data[:, 3]

    result_list.append(data[:, 3])
    fname = f"lpdsolver.dual_rates_time.n_sat{n}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    res[i, 2, :] = data[:, 3]

    fname = f"lpdsolver.dual_prices_time.n_sat{n}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    res[i, 3, :] = data[:, 3]

    fname = f"lpdsolver.prim_rounding_matching_time.n_sat{n}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    res[i, 4, :] = data[:, 3]

    fname = f"lpdsolver.prim_rounding_routing_time.n_sat{n}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    res[i, 5, :] = data[:, 3]

    fname = f"lpdsolver.prim_rounding_rates_time.n_sat{n}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    res[i, 6, :] = data[:, 3]

# colors = ["#FFD9B6","#FDBD88","#E96900","#AD4E00","#723300"]

markers = ['+', 's', 'D', '^', 'v', 'x', '*']
lines1 = []
lines2 = []
for j in range(7):    
    data  = res[:, j, :]
    print(data.shape)
    data_mean = np.mean(data, axis=1).reshape(-1)/1e6
    line, = axs.plot(np.array(n_sat), data_mean,linewidth=1,marker=markers[j], linestyle='-',markerfacecolor='None',markersize=5)
    lines1.append(line)

axs.set_position([0.15, 0.175, 0.8, 0.6])
# axs.set_title('Beam Intensity')
axs.set_xlabel(r'Number of Satellites, $I$')
axs.set_ylabel(r'Computing Time (s)')
# axs.set_xlim(1, N_POINTS)
axs.set_ylim(0.001, 0.2)
# axs.set_xscale('log')
axs.set_yscale('log')
axs.grid()
# Add a legend

data_name_list = ["d. MWM", "d. SPF","d. FRM", "d. SG",
                  "p. MWM","p. SPF","p. FRM"
                   ]
ncol = 4  # Number of columns in the legend
h, l = lines1, data_name_list
nrows = -(-len(h) // ncol)                         # ceiling division
idx   = np.arange(len(h))
pad   = nrows * ncol - len(idx)
idx   = np.concatenate([idx, np.full(pad, -1)])    # pad
order = idx.reshape(nrows, ncol).T.ravel()
order = order[order >= 0]                          # drop sentinels
h = [h[i] for i in order]
l = [l[i] for i in order]

leg = fig.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.15, 0.8, 0.8, 0.15), mode="expand",ncol = 4 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.4, 
frameon=True,          # draw a frame
fancybox=False,        # <-- square corners (like MATLAB)
edgecolor='black',     # black  frame edge
facecolor='white',     # white background
framealpha=1,          # fully opaque
borderpad=0.3,         # tight inner padding (font-size units)
labelspacing=0.2,      # tight vertical space between rows
)   
leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border
# axs.add_artist(leg)

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