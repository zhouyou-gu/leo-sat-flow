import os
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 500
dpi = 100  # Typical screen DPI, adjust if necessary
fig_width_in = fig_width_px / dpi
fig_height_in = fig_height_px / dpi

plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')
plt.rc('font', size=FONT_SIZE)  # Default font size
plt.rc('axes', titlesize=FONT_SIZE)  # Font size of the axes title
plt.rc('axes', labelsize=FONT_SIZE)  # Font size of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)  # Font size of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)  # Font size of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)  # Font size for legends

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='full')


from working_dir_path import get_working_dir_path
current_dir = os.path.dirname(os.path.abspath(__file__))

data_name_list = ["gap","rate","ratio to heuristic"]

# Create subplots
fig, axs = plt.subplots(3, 1)
# fig_width_in = (fig_width_px * 3 + 40) / dpi  # Adjust width for three subplots and spacing
fig.set_size_inches(fig_width_in, fig_height_in)

# Define the paths to your three data files
datas = [
    os.path.join(get_working_dir_path(), "sim_alg/test_lpd_lct4_dual_only/test_lpd_lct4_dual_only-2025-June-03-12-08-37-ail/DualSimulation.gap.lpd.txt"),
    os.path.join(get_working_dir_path(), "sim_alg/test_lpd_lct4_dual_only/test_lpd_lct4_dual_only-2025-June-03-12-08-37-ail/DualSimulation.p_o.lpd.txt"),
    os.path.join(get_working_dir_path(), "sim_alg/test_lpd_lct4_dual_only/test_lpd_lct4_dual_only-2025-June-03-12-08-37-ail/DualSimulation.ratio.lpd.txt"),
]

data_list = []
time_list = []
for idx, f in enumerate(datas):
    d = np.genfromtxt(f, delimiter=',')[:,3]
    data_list.append(d)

# rho_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lines = []
for idx, data in enumerate(data_list):
    # Plot the data
    line, = axs[idx].plot(data, linewidth=1.5, markerfacecolor='none')
    lines.append(line)
    axs[idx].grid()
    axs[idx].legend([data_name_list[idx]], fontsize=FONT_SIZE, loc='upper right', framealpha=1, handlelength=1.5, fancybox=True)


# # Add a legend
# fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.175, 0.915, 0.75, 0.1), ncol = 3 , borderaxespad=0.1,handlelength=1.5,fancybox=True, framealpha=1,mode='expand' )
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
# plt.show()