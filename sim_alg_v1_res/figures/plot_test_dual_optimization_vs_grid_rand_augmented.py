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


FIG_WIDTH_PX = 410
FIG_HEIGHT_PX = 245
DPI = 100
X_LABELS = ["500", "750", "1000", "1250", "1500", "1750", "2000"]


apply_plot_style()

current_dir = os.path.dirname(os.path.abspath(__file__))
results_file = latest_augmentation_file("mixed_constellation_merged.csv")
data = np.genfromtxt(results_file, delimiter=",")
data = -data[:, METHOD_COLUMN_INDICES]

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(FIG_WIDTH_PX / DPI, FIG_HEIGHT_PX / DPI)

bar_width = 0.22
bars = []
index = np.arange(data.shape[0]) * 2.0
for idx, method_name in enumerate(METHOD_NAMES):
    bar = axs.bar(
        index + (idx - (len(METHOD_NAMES) - 1) / 2.0) * bar_width,
        data[:, idx],
        bar_width,
        label=method_name,
        color=METHOD_COLORS[method_name],
        zorder=3,
    )
    bars.append(bar)

axs.set_position([0.15, 0.18, 0.82, 0.60])
axs.set_ylabel(r"Network Throughput (Gbps)")
axs.set_xlabel(r"Number of Satellites, $I$")
axs.set_xticks(index)
axs.set_xticklabels(X_LABELS)
axs.grid(True, zorder=0, alpha=0.35)

legend = fig.legend(
    bars,
    METHOD_NAMES,
    fontsize=FONT_SIZE,
    loc="lower left",
    bbox_to_anchor=(0.15, 0.82, 0.82, 0.16),
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
