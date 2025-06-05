import os
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 400
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

data_name_list = ["gap","rate","ratio to heuristic"]

# Create subplots
fig, axs = plt.subplots(1, 1)
# fig_width_in = (fig_width_px * 3 + 40) / dpi  # Adjust width for three subplots and spacing
fig.set_size_inches(fig_width_in, fig_height_in)

# Define the paths to your three data files
path = os.path.join(get_working_dir_path(), "demo/demo_4_jun_var_for/demo_4_jun_var_for-2025-June-04-17-30-33-ail")

data_list = []
for THETA in np.linspace(10, 90, 5):
    s = f"STATS_OBJECT.capacity_{int(THETA):02d}.final.txt"
    print(type(s), type(path))
    data_path = os.path.join(path, s)
    d = np.genfromtxt(data_path, delimiter=',')[3:]
    data_list.append(d)

lines = []
for idx, data in enumerate(data_list):
    # Plot the data
    data = np.sort(data, axis=0)
    print(data.shape)
    line, = axs.plot(data, linewidth=1.5, markerfacecolor='none')
    lines.append(line)

axs.grid()
axs.legend(lines, [f"FOR={int(THETA):02d} deg" for THETA in np.linspace(10, 90, 5)], fontsize=FONT_SIZE, loc='upper left', framealpha=1, handlelength=1.5, fancybox=True)


# add the arrow pointing to the middle of the plot
axs.annotate('Intra-orbit links \nwith the same distances', xy=(850, 5.5), xytext=(100, 8),
             arrowprops=dict(
        arrowstyle='->',             # arrow style
        lw=1.5,                      # line width of arrow
        color='black',               # arrow color
        shrinkA=0, shrinkB=5        # shorten arrow at tip and tail (in points)
    ))
axs.set_xlabel('Index of Sorted Links')
axs.set_ylabel('Capacity (Gbps)')
axs.set_title('Link Capacities in Regular Walker-Delta Constellation', fontsize=FONT_SIZE)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()