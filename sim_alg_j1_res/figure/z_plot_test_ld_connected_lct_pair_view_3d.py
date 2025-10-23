import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

#create a figure with (2,1) subplots
FONT_SIZE = 9
fig_width_px = 1200
fig_height_px = 600
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

results_dir = "/Users/charles_gu/Documents/GitHub/leo-sat-flow/sim_alg_j1_res/test_ld_connected_lct_pair_view/test_ld_connected_lct_pair_view-2025-October-23-19-26-10-ail"
# Plot the data
fig = plt.figure()
fig.set_size_inches(fig_width_in, fig_height_in)
# Create 3D subplots
axs_list = [fig.add_subplot(3, 4, i+1, projection='3d') for i in range(12)]  

result_list = []
FOR_list = [30, 60, 90]
METHOD_list = ['dujo','mwm', 'rand', 'grid']
lines1 = []
counter = 0
for i, FOR in enumerate(FOR_list):
    for j, METHOD in enumerate(METHOD_list):
        data_file = os.path.join(results_dir, f"relative_lct_view_for_{FOR}_seed_0_{METHOD}.csv")
        data = np.genfromtxt(data_file, delimiter=',')
        print(f"Data shape for FOR={FOR}, METHOD={METHOD}: {data.shape}")
        axs = axs_list[counter]
        # randomly select 100 points to plot
        # if data.shape[0] > 100:
        #     indices = np.random.choice(data.shape[0], size=100, replace=False)
        #     data = data[indices, :]
        #3d scatter plot
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]
        xyz = np.vstack([x, y, z])
        density = gaussian_kde(xyz)(xyz)

        # Sort the points by density (for better plotting)
        idx = density.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        l = axs.scatter(x, y, z, c=density[idx], alpha=0.1, label=f"FOR {FOR} - {METHOD}", s=0.2, cmap='viridis')
        lines1.append(l)

        axs_list[counter].set_xlim(0, 3000)
        axs_list[counter].set_ylim(-3000, 3000)
        axs_list[counter].set_zlim(-3000, 3000)
        axs_list[counter].set_xlabel('X')
        axs_list[counter].set_ylabel('Y')
        axs_list[counter].set_zlabel('Z')
        axs_list[counter].set_title(f'FOR = {FOR}')
        # set equal aspect ratio
        axs_list[counter].set_box_aspect([0.5,1,1])  # Aspect 
        
        counter += 1

# axs.set_position([0.18, 0.2, 0.3, 0.765])
# axs.set_xlabel(r'Number of Iterations, $K$')
# axs.set_ylabel(r'Dual Function Value, $g(\lambda)$')
# axs.set_xlim(0, N_POINTS)
# axs.set_ylim(-3000, -0)
# axs.grid(True, zorder=0)
# axs.text(20, -5000, r'$\bf{(a)}$', fontsize=FONT_SIZE+2, verticalalignment='bottom')



data_name_list = [r"LaDuGL", r"DPG", r"PG"] 
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

leg = axs.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.35, 0.025, 0.6, 0.3), mode="expand",ncol = 1 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.5, 
frameon=True,          # draw a frame
fancybox=False,        # <-- square corners (like MATLAB)
edgecolor='black',     # black  frame edge
facecolor='white',     # white background
framealpha=1,          # fully opaque
borderpad=0.3,         # tight inner padding (font-size units)
labelspacing=0.2,      # tight vertical space between rows
)   
leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border

plt.show()
# # Save the figure as a PDF
# print("Saving figure to:", current_dir)
# output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

# fig.savefig(output_path, format='pdf', pad_inches=0.)