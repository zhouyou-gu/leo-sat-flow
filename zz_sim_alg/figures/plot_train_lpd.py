import os
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 350
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

data_name_list = ["LPD","MDM","SPF"]

# Create subplots
fig, axs = plt.subplots(1, 1)
# fig_width_in = (fig_width_px * 3 + 40) / dpi  # Adjust width for three subplots and spacing
fig.set_size_inches(fig_width_in, fig_height_in)

# Define the paths to your three data files
folder = [
    os.path.join(get_working_dir_path(), "sim_alg/train_lpd/train_lpd-2025-May-26-11-06-18-ail"),
    os.path.join(get_working_dir_path(), "sim_alg/train_lpd/train_lpd-2025-May-27-10-45-07-ail"),
    os.path.join(get_working_dir_path(), "sim_alg/train_lpd/train_lpd-2025-May-27-11-48-34-ail"),
]

data_list = []
time_list = []
for idx, f in enumerate(folder):
    thr_file = os.path.join(f,"DualSimulation.objective.final.txt")
    thr_lpd = np.genfromtxt(thr_file, delimiter=',')[:,3]
    thr_mwm = np.genfromtxt(thr_file, delimiter=',')[:,4]
    thr = thr_lpd/thr_mwm
    # thr = np.mean(thr.reshape(-1, 10), axis=1)
    data_list.append(thr)

# rho_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lines = []
for idx, data in enumerate(data_list):
    # Plot the data
    line, = axs.plot(data, linewidth=1.5, markerfacecolor='none')
    lines.append(line)

# Add a legend
fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.175, 0.915, 0.75, 0.1), ncol = 3 , borderaxespad=0.1,handlelength=1.5,fancybox=True, framealpha=1,mode='expand' )
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)

axs.grid()

# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
# plt.show()