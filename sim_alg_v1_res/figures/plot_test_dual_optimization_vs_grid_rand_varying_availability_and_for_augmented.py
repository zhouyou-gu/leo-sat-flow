import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

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
FOR_LABELS = ["30", "50", "70", "90"]
LCT_AVAILABILITY_LABELS = ["0.8", "1.2", "1.6", "2.0"]


apply_plot_style()
FONT_SIZE = 10
plt.rcParams.update({key: FONT_SIZE for key in ["font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"]})

current_dir = os.path.dirname(os.path.abspath(__file__))

# Recorded final results recovered from April 10 tool output, six decimals.
# Keep all source rows; display four evenly spaced cases in each panel.
data_dir = Path(current_dir).parent / "recovered_legacy_results/recovered-20260923-ail"
for_data = -pd.read_csv(data_dir / "varying_for.csv")[METHOD_NAMES].to_numpy()[::2]
availability_data = -pd.read_csv(data_dir / "lct_failure_rate.csv")[METHOD_NAMES].to_numpy()[::-1][::2]

fig, axs_list = plt.subplots(1, 2)
fig.set_size_inches(FIG_WIDTH_PX / DPI, FIG_HEIGHT_PX / DPI)

bar_width = 0.28

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
axs.set_xlabel(r"Average LCTs per Sat.")
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
axs.set_xlabel(r"Field of Regard Size (deg)")
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

output_path = os.path.join(current_dir, "plot_test_dual_optimization_vs_grid_rand_varying_availability_and_for.pdf")
fig.savefig(output_path, format="pdf", pad_inches=0.0)
