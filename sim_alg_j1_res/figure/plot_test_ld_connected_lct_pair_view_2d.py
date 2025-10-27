import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

#create a figure with (2,1) subplots
FONT_SIZE = 9
fig_width_px = 900
fig_height_px = 360
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
axs_list = [fig.add_subplot(3, 4, i+1) for i in range(12)]  # Create 2D subplots

#generate a list of subplot positions in a 3x4 grid of the figure's relative coordinates
horizontal_gap = 0.01
vertical_gap = 0.03
# Calculate subplot dimensions based on figure size and gaps, 
# ensuring X:Y aspect ratio is 2:1
subplot_width = (1 - 5 * horizontal_gap) / 4 - 0.03
subplot_height = subplot_width / 2 * (fig_width_in / fig_height_in)
horizontal_offset = 0.08
vertical_offset = 0.02257

subplot_positions = [ ( (i % 4) * (subplot_width + horizontal_gap) + horizontal_offset, 1 - (i // 4 + 1) * (subplot_height + vertical_gap) - vertical_offset, subplot_width , subplot_height ) for i in range(12) ]

result_list = []
FOR_list = [30, 60, 90]
METHOD_list = ['dujo','mwm', 'rand', 'grid']
METHOD_LABELS = {
    'dujo': 'LaDuGL',
    'mwm': 'MRate',
    'rand': 'Rand',
    'grid': '+Grid'
}
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
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density (for better plotting)
        idx = z.argsort()
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        z = z*1e6
        print(f"Z min: {np.min(z)}, Z max: {np.max(z)}")
        l = axs.scatter(y, x, c=z, alpha=1, label=f"FOR {FOR} - {METHOD}", s=5, cmap='viridis', vmax=1, vmin=0, zorder=3)
        lines1.append(l)
        axs.grid(True, zorder=0)

        axs_list[counter].set_xlim(-3000, 3000)
        axs_list[counter].set_ylim(0, 3000)
        axs_list[counter].set_position(subplot_positions[counter])
        # axs_list[counter].set_aspect('equal', adjustable='box')
        axs_list[counter].set_xticks([-3000, -2000, -1000, 0, 1000, 2000, 3000])
        axs_list[counter].set_yticks([0, 1000, 2000, 3000])
        axs_list[counter].set_xticklabels([])
        axs_list[counter].set_yticklabels([])
        if counter <= 3:
            axs_list[counter].set_title(f'{METHOD_LABELS[METHOD]}', fontsize=FONT_SIZE)
        if counter % 4 == 0:
            axs_list[counter].set_ylabel('Y (km)', fontsize=FONT_SIZE)
            axs_list[counter].set_yticklabels([0, 1000, 2000, 3000])
            # title on y-axis
            axs_list[counter].text(-0.35, 0.5, f'FOR={FOR}°', va='center', ha='center', rotation='vertical', fontsize=FONT_SIZE, transform=axs_list[counter].transAxes)
        if counter >= 8:
            axs_list[counter].set_xlabel('X (km)', fontsize=FONT_SIZE)
            axs_list[counter].set_xticklabels([None, -2000, -1000, 0, 1000, 2000, None])
        print(axs_list[counter].get_position())
        counter += 1


# Draw colorbar at the bottom of the figure, in horizontal orientation
cbar_ax = fig.add_axes([0.945, 0.109305, 0.01, 0.94743-0.109305])  # [left, bottom, width, height]
cbar = fig.colorbar(lines1[0], cax=cbar_ax)
# Set ticks and tick labels
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels([r'0', r'0.5', r'$\geq 1.0$'], fontsize=FONT_SIZE)
cbar.set_label('Point Density')
fig.text(0.97, 0.15, 'Point Density (per km$^2$)', va='bottom', ha='left', rotation='vertical',
         fontsize=FONT_SIZE, horizontalalignment='center')

# data_name_list = [r"LaDuGL", r"DPG", r"PG"] 
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

# leg = axs.legend(h,l,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.35, 0.025, 0.6, 0.3), mode="expand",ncol = 1 ,borderaxespad=0.,handlelength=1, handleheight= 0.8, handletextpad=0.5, 
# frameon=True,          # draw a frame
# fancybox=False,        # <-- square corners (like MATLAB)
# edgecolor='black',     # black  frame edge
# facecolor='white',     # white background
# framealpha=1,          # fully opaque
# borderpad=0.3,         # tight inner padding (font-size units)
# labelspacing=0.2,      # tight vertical space between rows
# )   
# leg.get_frame().set_linewidth(0.8)  # MATLAB-thin border

# Save the figure as a PDF
print("Saving figure to:", current_dir)
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)