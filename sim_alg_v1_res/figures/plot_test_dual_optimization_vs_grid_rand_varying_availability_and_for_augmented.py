import os

import matplotlib.pyplot as plt
import numpy as np

from legacy_augmented_plot_common import (
    FONT_SIZE,
    METHOD_COLORS,
    METHOD_COLUMN_INDICES,
    METHOD_NAMES,
    apply_plot_style,
    latest_augmentation_file,
)


FIG_WIDTH_PX = 400
FIG_HEIGHT_PX = 255
DPI = 100
FOR_LABELS = ["30", "40", "50", "60", "70", "80", "90"]
LCT_AVAILABILITY_LABELS = ["0.8", "1.0", "1.2", "1.4", "1.6", "1.8", "2.0"]


apply_plot_style()

current_dir = os.path.dirname(os.path.abspath(__file__))

for_data = np.genfromtxt(latest_augmentation_file("varying_for_merged.csv"), delimiter=",")
for_data = -for_data[:, METHOD_COLUMN_INDICES]

availability_data = np.genfromtxt(
    latest_augmentation_file("lct_failure_rate_merged.csv"),
    delimiter=",",
)
availability_data = -availability_data[:, METHOD_COLUMN_INDICES]
availability_data = availability_data[::-1, :]

fig, axs_list = plt.subplots(1, 2)
fig.set_size_inches(FIG_WIDTH_PX / DPI, FIG_HEIGHT_PX / DPI)

bar_width = 0.22

axs = axs_list[0]
bars = []
index = np.arange(availability_data.shape[0]) * 2.0
for idx, method_name in enumerate(METHOD_NAMES):
    bar = axs.bar(
        index + (idx - (len(METHOD_NAMES) - 1) / 2.0) * bar_width,
        availability_data[:, idx],
        bar_width,
        label=method_name,
        color=METHOD_COLORS[method_name],
        zorder=3,
    )
    bars.append(bar)
axs.set_position([0.15, 0.19, 0.37, 0.57])
axs.set_ylabel(r"Network Throughput (Gbps)")
axs.set_xlabel(r"Average Number of LCTs per Sat.")
axs.set_xticks(index)
axs.set_xticklabels(LCT_AVAILABILITY_LABELS)
axs.grid(True, zorder=0, alpha=0.35)

axs = axs_list[1]
index = np.arange(for_data.shape[0]) * 2.0
for idx, method_name in enumerate(METHOD_NAMES):
    axs.bar(
        index + (idx - (len(METHOD_NAMES) - 1) / 2.0) * bar_width,
        for_data[:, idx],
        bar_width,
        label=method_name,
        color=METHOD_COLORS[method_name],
        zorder=3,
    )
axs.set_position([0.575, 0.19, 0.37, 0.57])
axs.set_yticklabels([])
axs.set_xlabel(r"Field of Regard Size (Degree)")
axs.set_xticks(index)
axs.set_xticklabels(FOR_LABELS)
axs.grid(True, zorder=0, alpha=0.35)

legend = fig.legend(
    bars,
    METHOD_NAMES,
    fontsize=FONT_SIZE,
    loc="lower left",
    bbox_to_anchor=(0.15, 0.81, 0.795, 0.16),
    mode="expand",
    ncol=3,
    borderaxespad=0.0,
    handlelength=1,
    handleheight=0.8,
    handletextpad=0.2,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1,
    borderpad=0.3,
    labelspacing=0.2,
)
legend.get_frame().set_linewidth(0.8)

output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0] + ".pdf")
fig.savefig(output_path, format="pdf", pad_inches=0.0)
