import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from legacy_augmented_plot_common import FONT_SIZE, METHOD_COLORS, apply_plot_style
from tcom_revision_common import HEADLINE_METHODS, latest_run_dir


FIG_WIDTH_PX = 350
FIG_HEIGHT_PX = 285
DPI = 100
FIG_WIDTH_IN = FIG_WIDTH_PX / DPI
FIG_HEIGHT_IN = FIG_HEIGHT_PX / DPI
LEGEND_BOX = (0.15, 0.915, 0.825, 0.12)
TOP_AXIS_BOX = [0.15, 0.55, 0.825, 0.35]
BOTTOM_AXIS_BOX = [0.15, 0.15, 0.825, 0.35]

apply_plot_style()

METHOD_STYLE = {
    "DuJo": {"color": METHOD_COLORS["DuJo"], "marker": "o"},
    "DRL": {"color": METHOD_COLORS["DRL"], "marker": "s"},
    "SaTE": {"color": METHOD_COLORS["SaTE"], "marker": "^"},
    "MRate": {"color": METHOD_COLORS["MRate"], "marker": "D"},
    "+Grid": {"color": METHOD_COLORS["+Grid"], "marker": "v"},
}

METHOD_ALIASES = {
    "DRL": ["DRL", "LD"],
}

PLOT_METHOD_SET = set()
for method in HEADLINE_METHODS:
    PLOT_METHOD_SET.add(method)
    PLOT_METHOD_SET.update(METHOD_ALIASES.get(method, [method]))

run_parent = os.path.join(PARENT_DIR, "test_tcom_shell_time_baselines")
run_dir = os.getenv("TCOM_REVISION_RESULTS_DIR", "").strip()
if not run_dir:
    run_dir = latest_run_dir(run_parent)
results_path = os.path.join(run_dir, "results.csv")
print("Using results from:", results_path)

df = pd.read_csv(results_path)
df = df[df["method"].isin(PLOT_METHOD_SET)].copy()
active_user_filter = os.getenv("TCOM_REVISION_ACTIVE_USER_PERCENTAGE", "").strip()
if active_user_filter:
    target_percentage = float(active_user_filter)
    df = df[df["active_user_percentage"] == target_percentage].copy()
df.sort_values(["method", "offset_minutes"], inplace=True)

fig, axs = plt.subplots(2, 1)
fig.set_size_inches(FIG_WIDTH_IN, FIG_HEIGHT_IN)

for method in HEADLINE_METHODS:
    method_df = df[df["method"].isin(METHOD_ALIASES.get(method, [method]))]
    style = METHOD_STYLE[method]
    axs[0].plot(
        method_df["offset_minutes"],
        method_df["served_throughput_gbps"],
        label=method,
        linewidth=1.2,
        marker=style["marker"],
        markersize=4,
        color=style["color"],
    )
    axs[1].plot(
        method_df["offset_minutes"],
        method_df["served_ratio"],
        label=method,
        linewidth=1.2,
        marker=style["marker"],
        markersize=4,
        color=style["color"],
    )

axs[0].set_ylabel("Throughput (Gbps)")
axs[1].set_ylabel("Served Ratio")
axs[1].set_xlabel("Time Offset from $T_0$ (min)")
axs[0].grid(True, zorder=0, alpha=0.35, linewidth=0.5)
axs[1].grid(True, zorder=0, alpha=0.35, linewidth=0.5)

for ax in axs:
    ax.set_xticks(sorted(df["offset_minutes"].unique()))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="y", labelsize=FONT_SIZE - 1)

axs[0].tick_params(labelbottom=False)

throughput_min = df["served_throughput_gbps"].min()
throughput_max = df["served_throughput_gbps"].max()
throughput_margin = 0.08 * (throughput_max - throughput_min)
axs[0].set_ylim(throughput_min - throughput_margin, throughput_max + throughput_margin)

ratio_min = df["served_ratio"].min()
ratio_max = df["served_ratio"].max()
ratio_margin = 0.10 * (ratio_max - ratio_min)
axs[1].set_ylim(max(0.0, ratio_min - ratio_margin), min(1.0, ratio_max + ratio_margin))

axs[0].set_position(TOP_AXIS_BOX)
axs[1].set_position(BOTTOM_AXIS_BOX)
handles, labels = axs[0].get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    fontsize=FONT_SIZE,
    loc="lower left",
    bbox_to_anchor=LEGEND_BOX,
    mode="expand",
    ncol=5,
    borderaxespad=0.0,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1,
    borderpad=0.3,
    labelspacing=0.2,
    handlelength=1,
    handleheight=0.8,
    handletextpad=0.2,
)
legend.get_frame().set_linewidth(0.8)

output_path = os.path.join(CURRENT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".pdf")
fig.savefig(output_path, format="pdf", pad_inches=0.0)
