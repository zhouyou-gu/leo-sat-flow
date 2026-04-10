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

run_parent = os.path.join(PARENT_DIR, "test_tcom_shell_time_load_scaling")
run_dir = os.getenv("TCOM_REVISION_RESULTS_DIR", "").strip()
if not run_dir:
    run_dir = latest_run_dir(run_parent)
results_path = os.path.join(run_dir, "results.csv")
print("Using results from:", results_path)

df = pd.read_csv(results_path)
df = df[df["method"].isin(PLOT_METHOD_SET)].copy()
agg = (
    df.groupby(["active_user_percentage", "method"], as_index=False)[
        ["served_throughput_gbps", "served_ratio"]
    ]
    .mean()
)
agg["active_user_percentage_pct"] = agg["active_user_percentage"] * 100.0
x_levels = sorted(agg["active_user_percentage_pct"].unique())
x_positions = list(range(len(x_levels)))
x_pos_map = dict(zip(x_levels, x_positions))

fig, axs = plt.subplots(2, 1)
fig.set_size_inches(FIG_WIDTH_IN, FIG_HEIGHT_IN)

for method in HEADLINE_METHODS:
    method_df = agg[agg["method"].isin(METHOD_ALIASES.get(method, [method]))].sort_values(
        "active_user_percentage_pct"
    )
    style = METHOD_STYLE[method]
    axs[0].plot(
        method_df["active_user_percentage_pct"].map(x_pos_map),
        method_df["served_throughput_gbps"],
        label=method,
        linewidth=1.2,
        marker=style["marker"],
        markersize=4,
        color=style["color"],
    )
    axs[1].plot(
        method_df["active_user_percentage_pct"].map(x_pos_map),
        method_df["served_ratio"],
        label=method,
        linewidth=1.2,
        marker=style["marker"],
        markersize=4,
        color=style["color"],
    )

axs[0].set_ylabel("Throughput (Gbps)")
axs[1].set_ylabel("Served Ratio")
axs[1].set_xlabel("Active User Percentage (%)", labelpad=0)
axs[0].grid(True, zorder=0, alpha=0.35, linewidth=0.5)
axs[1].grid(True, zorder=0, alpha=0.35, linewidth=0.5)

for ax in axs:
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{tick:.4f}".rstrip("0").rstrip(".") for tick in x_levels])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="y", labelsize=FONT_SIZE - 1)

axs[1].tick_params(axis="x", labelsize=FONT_SIZE - 1, rotation=15)
for label in axs[1].get_xticklabels():
    label.set_horizontalalignment("right")

axs[0].tick_params(labelbottom=False)

axs[0].set_ylim(0.0, 450.0)
axs[0].set_yticks([0, 150, 300, 450])
ratio_min = agg["served_ratio"].min()
ratio_max = agg["served_ratio"].max()
ratio_margin = 0.08 * (ratio_max - ratio_min)
axs[1].set_ylim(max(0.0, ratio_min - ratio_margin), min(1.0, ratio_max + ratio_margin))
axs[1].set_yticks([0.0, 0.15, 0.30, 0.45])
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
