"""Plot the steady-load sweep as two side-by-side line charts."""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from sim_alg_j1_res.varying_load_helpers import (
    demand_satisfaction_percentage,
    traffic_demand_rate_series,
)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
RUN_PARENT = os.path.join(PARENT_DIR, "test_ld_starlink_1000_varying_load")
FIG_WIDTH_PX = 400
FIG_HEIGHT_PX = 225
DPI = 100
FONT_SIZE = 9
LEGEND_BOX = (0.125, 0.9, 0.865, 0.125)
LEFT_AXIS_BOX = [0.125, 0.18, 0.38, 0.69]
RIGHT_AXIS_BOX = [0.61, 0.18, 0.38, 0.69]

METHOD_ORDER = ["DeepLaDu", "Rand", "+Grid", "MRate", "SaTE"]
METHOD_LABEL = {
    "DeepLaDu": "DeepLaDu",
    "Rand": "Random",
    "+Grid": "+Grid",
    "MRate": "MRate",
    "SaTE": "SaTE",
}
METHOD_STYLE = {
    "DeepLaDu": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "Rand": {"color": "#ff7f0e", "marker": "P", "linestyle": ":"},
    "+Grid": {"color": "#2ca02c", "marker": "v", "linestyle": "-."},
    "MRate": {"color": "#d62728", "marker": "D", "linestyle": "--"},
    "SaTE": {"color": "#9467bd", "marker": "^", "linestyle": (0, (3, 1, 1, 1))},
}


def latest_run_dir(parent_dir):
    candidates = [
        os.path.join(parent_dir, entry)
        for entry in os.listdir(parent_dir)
        if os.path.isfile(os.path.join(parent_dir, entry, "summary.csv"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete load-sweep run found under {parent_dir}")
    return max(candidates, key=os.path.getmtime)


def apply_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "STIXGeneral",
            ],
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plt.rc("mathtext", fontset="cm")


def main():
    apply_plot_style()
    run_dir = os.getenv("DEEP_LADU_LOAD_RESULTS_DIR", "").strip()
    if not run_dir:
        run_dir = latest_run_dir(RUN_PARENT)
    summary_path = os.path.join(run_dir, "summary.csv")
    print("Using summary from:", summary_path)

    summary = pd.read_csv(summary_path)
    summary = summary[summary["method"].isin(METHOD_ORDER)].copy()
    methods = set(summary["method"].unique())
    if methods != set(METHOD_ORDER):
        raise ValueError(f"Expected methods {METHOD_ORDER}, found {sorted(methods)}")

    summary["active_user_percentage_pct"] = (
        summary["active_user_percentage"] * 100.0
    )
    x_levels = sorted(summary["active_user_percentage_pct"].unique())
    x_positions = list(range(len(x_levels)))
    x_pos_map = dict(zip(x_levels, x_positions))

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(FIG_WIDTH_PX / DPI, FIG_HEIGHT_PX / DPI)

    demand_series = traffic_demand_rate_series(summary.to_dict("records"))
    demand_loads = [load * 100.0 for load, _ in demand_series]
    demand_x_values = [x_pos_map[load] for load in demand_loads]
    demand_values = [demand for _, demand in demand_series]
    axs[0].plot(
        demand_x_values,
        demand_values,
        linewidth=1.2,
        marker="o",
        markersize=3.7,
        color="#4d4d4d",
    )

    for method in METHOD_ORDER:
        method_df = summary[summary["method"] == method].sort_values(
            "active_user_percentage_pct"
        )
        if len(method_df) != len(x_levels):
            raise ValueError(f"Incomplete load series for {method}")
        style = METHOD_STYLE[method]
        x_values = method_df["active_user_percentage_pct"].map(x_pos_map)
        common = {
            "label": METHOD_LABEL[method],
            "linewidth": 1.2,
            "marker": style["marker"],
            "markersize": 3.7,
            "color": style["color"],
            "linestyle": style["linestyle"],
        }
        axs[1].plot(
            x_values,
            method_df["served_ratio_mean"].map(
                demand_satisfaction_percentage
            ),
            **common,
        )

    axs[0].set_ylabel("Traffic demand rate (Gbps)")
    axs[1].set_ylabel("Demand satisfaction ratio (%)")
    for ax in axs:
        ax.set_xlabel("Active User Percentage (%)", labelpad=1)

    labeled_tick_indices = [0, 2, 4, len(x_levels) - 1]
    for ax in axs:
        ax.grid(True, zorder=0, alpha=0.35, linewidth=0.5)
        ax.set_xticks(labeled_tick_indices)
        ax.set_xticklabels(
            [
                f"{x_levels[index]:.4f}".rstrip("0").rstrip(".")
                for index in labeled_tick_indices
            ]
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="y", labelsize=FONT_SIZE - 1)
        ax.tick_params(axis="x", labelsize=FONT_SIZE - 1)

    demand_upper = max(500.0, math.ceil(max(demand_values) * 1.08 / 500.0) * 500.0)
    served_max = demand_satisfaction_percentage(
        summary["served_ratio_mean"].max()
    )
    served_upper = max(10.0, math.ceil(served_max * 1.08 / 10.0) * 10.0)
    axs[0].set_ylim(0.0, demand_upper)
    axs[1].set_ylim(0.0, served_upper)
    axs[0].set_position(LEFT_AXIS_BOX)
    axs[1].set_position(RIGHT_AXIS_BOX)

    handles, labels = axs[1].get_legend_handles_labels()
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

    output_path = os.path.join(
        CURRENT_DIR,
        os.path.splitext(os.path.basename(__file__))[0] + ".pdf",
    )
    fig.savefig(output_path, format="pdf", pad_inches=0.0)
    print("Wrote:", output_path)


if __name__ == "__main__":
    main()
