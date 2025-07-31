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


current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, os.path.pardir, "test_dual_optimization_starlink_1000_varying_beta")

# Plot the data
fig, axs_list = plt.subplots(1,2,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

beta = [0.5, 0.7, 0.9]
N_POINTS = 2000
res = np.zeros((len(beta), 4, N_POINTS))
result_list = []
for i, b in enumerate(beta):
    print(f"Processing beta = {b}")
    tmp_dir = "test_dual_optimization_starlink_1000_varying_beta-2025-July-26-13-24-29-ail"
    fname = f"DualSimulation.d_o.beta{int(b*10)}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    
    res[i, 0, :] = data[:, 3]
    result_list.append(data[:, 3])
    fname = f"DualSimulation.gap.beta{int(b*10)}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    res[i, 1, :] = data[:, 3]

    fname = f"DualSimulation.mwm.beta{int(b*10)}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    res[i, 2, :] = data[:, 3]
    
    fname = f"DualSimulation.p_o.beta{int(b*10)}.txt"
    data_file = os.path.join(results_dir, tmp_dir, fname)
    data = np.genfromtxt(data_file, delimiter=',')
    res[i, 3, :] = data[:, 3]

# colors = ["#FFD9B6","#FDBD88","#E96900","#AD4E00","#723300"]
axs = axs_list[0]
lines1 = []
lines2 = []
for i, b in enumerate(beta):
    print(f"Plotting beta = {b}",i)
    print(result_list)
    line, = axs.plot(np.arange(N_POINTS)+1, res[i,0,:],linewidth=1, zorder=3)
    lines1.append(line)
    # line, = axs.plot(np.arange(N_POINTS)+1, res[i,1,:],linewidth=1)
    # lines2.append(line)
    # line, = axs.plot(np.arange(N_POINTS)+1, res[i,2,:],linewidth=1)
    # lines.append(line)
    # line, = axs.plot(np.arange(N_POINTS)+1, res[i,3,:],linewidth=1)
    # lines.append(line)

axs.set_position([0.18, 0.2, 0.3, 0.765])
# axs.set_title('Beam Intensity')
axs.set_xlabel(r'Number of Iterations, $K$')
axs.set_ylabel(r'Dual Function Value, $g(\lambda)$')
axs.set_xlim(0, N_POINTS)
axs.set_ylim(-1200, -400)
# axs.set_xscale('log')
# axs.set_yscale('log')
axs.grid(True, zorder=0)
axs.text(20, -1200, r'$\bf{(a)}$', fontsize=FONT_SIZE+2, verticalalignment='bottom')


# Add a legend

data_name_list = [r"$\beta=$"+f"{i}" for i in beta] 
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

axs = axs_list[1]
lines1 = []
lines2 = []
for i, b in enumerate(beta):
    print(f"Plotting beta = {b}",i)
    print(result_list)
    AVG_N = 5
    avg = np.convolve(-res[i,3,:], np.ones(AVG_N)/AVG_N, mode='full')[:N_POINTS]
    line, = axs.plot(np.arange(N_POINTS)+1, avg, linewidth=1., zorder=3)
    lines1.append(line)
    # line, = axs.plot(np.arange(N_POINTS)+1, res[i,1,:],linewidth=1)
    # lines2.append(line)
    # line, = axs.plot(np.arange(N_POINTS)+1, res[i,2,:],linewidth=1)
    # lines.append(line)
    # line, = axs.plot(np.arange(N_POINTS)+1, res[i,3,:],linewidth=1)
    # lines.append(line)
line, = axs.plot(np.arange(N_POINTS)+1, -res[0,2,:],linewidth=1)
lines1.append(line)

axs.set_position([0.63, 0.2, 0.33, 0.765])
# axs.set_title('Beam Intensity')
axs.set_xlabel(r'Number of Iterations, $K$')
axs.set_ylabel(r"Network Throughput (Gbps)")
axs.set_xlim(0, N_POINTS)
axs.set_ylim(0, 180)
# axs.set_xscale('log')
# axs.set_yscale('log')
axs.grid(True, zorder=0)
axs.text(20, 0, r'$\bf{(b)}$', fontsize=FONT_SIZE+2, verticalalignment='bottom')


data_name_list = [r"$\beta=$"+f"{i}" for i in beta] 
data_name_list.append(r"MRate")
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
axs.add_artist(leg)

# Save the figure as a PDF
print("Saving figure to:", current_dir)
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)