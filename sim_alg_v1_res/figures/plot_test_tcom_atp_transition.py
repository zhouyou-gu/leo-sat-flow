import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from legacy_augmented_plot_common import FONT_SIZE, METHOD_COLORS, apply_plot_style
from tcom_revision_common import (
    DEFAULT_ATP_TIME_SWEEP_SEC,
    HEADLINE_METHODS,
    SUPPORTING_METHODS,
    latest_run_dir,
)


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
    "Rand": {"color": METHOD_COLORS["Rand"], "marker": "x"},
}

METHOD_ALIASES = {
    "DRL": ["DRL", "LD"],
}

PLOT_METHOD_SET = set()
for method in HEADLINE_METHODS:
    PLOT_METHOD_SET.add(method)
    PLOT_METHOD_SET.update(METHOD_ALIASES.get(method, [method]))
for method in SUPPORTING_METHODS:
    PLOT_METHOD_SET.add(method)
    PLOT_METHOD_SET.update(METHOD_ALIASES.get(method, [method]))

run_parent = os.path.join(PARENT_DIR, "test_tcom_atp_transition")
run_dir = os.getenv("TCOM_REVISION_RESULTS_DIR", "").strip()
if not run_dir:
    run_dir = latest_run_dir(run_parent)
results_path = os.path.join(run_dir, "results.csv")
summary_path = os.path.join(run_dir, "summary.csv")
print("Using results from:", run_dir)

if os.path.exists(summary_path):
    df = pd.read_csv(summary_path)
else:
    results_df = pd.read_csv(results_path)
    if "atp_time_seconds" not in results_df.columns:
        raise ValueError("ATP-time sweep results require the atp_time_seconds column.")
    if "initialization_snapshot" in results_df.columns:
        results_df = results_df[results_df["initialization_snapshot"] == 0].copy()
    results_df["usable_link_fraction"] = results_df.apply(
        lambda row: 0.0 if row["matched_links"] == 0 else row["usable_links"] / row["matched_links"],
        axis=1,
    )
    df = (
        results_df.groupby(["method", "atp_time_seconds"], as_index=False)
        .agg(
            snapshot_count=("offset_seconds", "count"),
            avg_original_served_throughput_gbps=("original_served_throughput_gbps", "mean"),
            avg_atp_served_throughput_gbps=("atp_served_throughput_gbps", "mean"),
            avg_original_served_ratio=("original_served_ratio", "mean"),
            avg_atp_served_ratio=("atp_served_ratio", "mean"),
            avg_usable_link_fraction=("usable_link_fraction", "mean"),
            avg_new_link_fraction=("new_link_fraction", "mean"),
            avg_acquiring_links=("acquiring_links", "mean"),
        )
    )
    df["throughput_retention_ratio"] = (
        df["avg_atp_served_throughput_gbps"] / df["avg_original_served_throughput_gbps"]
    )

df = df[
    df["method"].isin(PLOT_METHOD_SET)
    & df["atp_time_seconds"].isin(DEFAULT_ATP_TIME_SWEEP_SEC)
].copy()
active_user_filter = os.getenv("TCOM_REVISION_ACTIVE_USER_PERCENTAGE", "").strip()
if active_user_filter:
    print("Ignoring TCOM_REVISION_ACTIVE_USER_PERCENTAGE for pre-aggregated ATP sweep plot.")
if df.empty:
    raise ValueError("No ATP transition summary rows are available for plotting after filtering.")
df.sort_values(["method", "atp_time_seconds"], inplace=True)

fig, axs = plt.subplots(2, 1)
fig.set_size_inches(FIG_WIDTH_IN, FIG_HEIGHT_IN)

plot_methods = [
    method
    for method in HEADLINE_METHODS + SUPPORTING_METHODS
    if method in set(df["method"])
]
for method in plot_methods:
    method_df = df[df["method"].isin(METHOD_ALIASES.get(method, [method]))]
    if method_df.empty:
        continue
    style = METHOD_STYLE[method]
    axs[0].plot(
        method_df["atp_time_seconds"],
        method_df["avg_atp_served_throughput_gbps"],
        label=method,
        linewidth=1.2,
        marker=style["marker"],
        markersize=4,
        color=style["color"],
        markerfacecolor="none",
        markeredgecolor=style["color"],
        markeredgewidth=1.0,
    )
    axs[1].plot(
        method_df["atp_time_seconds"],
        method_df["throughput_retention_ratio"],
        label=method,
        linewidth=1.2,
        marker=style["marker"],
        markersize=4,
        color=style["color"],
        markerfacecolor="none",
        markeredgecolor=style["color"],
        markeredgewidth=1.0,
    )

axs[0].set_ylabel("Throughput (Gbps)")
axs[1].set_ylabel("Retention Ratio")
axs[1].set_xlabel(r"ATP Acquisition Time, $T_{\mathrm{ATP}}$ (s)")
axs[0].grid(True, zorder=0, alpha=0.35, linewidth=0.5)
axs[1].grid(True, zorder=0, alpha=0.35, linewidth=0.5)

for ax in axs:
    ax.set_xticks(DEFAULT_ATP_TIME_SWEEP_SEC)
    ax.set_xlim(min(DEFAULT_ATP_TIME_SWEEP_SEC) - 1, max(DEFAULT_ATP_TIME_SWEEP_SEC) + 1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=FONT_SIZE - 1)

axs[0].tick_params(labelbottom=False)
axs[0].set_ylim(0.0, 150.0)
axs[0].set_yticks([0, 50, 100, 150])
axs[1].set_ylim(0.0, 1.0)
axs[1].set_yticks([0.0, 0.5, 1.0])

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
    ncol=min(5, max(1, len(labels))),
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
