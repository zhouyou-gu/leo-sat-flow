import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FONT_SIZE = 9
FIG_WIDTH_IN = 4.0
FIG_HEIGHT_IN = 2.25

METHOD_NAMES = ["DeepLaDu", "Rand", "+Grid", "MRate", "SaTE"]
METHOD_STYLES = [
    {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    {"color": "#ff7f0e", "marker": "P", "linestyle": ":"},
    {"color": "#2ca02c", "marker": "v", "linestyle": "-."},
    {"color": "#d62728", "marker": "D", "linestyle": "--"},
    {"color": "#9467bd", "marker": "^", "linestyle": (0, (3, 1, 1, 1))},
]


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "STIXGeneral",
            ],
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_lct_summary(summary_path):
    summary_path = Path(summary_path)
    rows = {}
    with summary_path.open(newline="", encoding="utf-8") as summary_file:
        for row in csv.DictReader(summary_file):
            level = float(row["average_lcts"])
            method = row["method"]
            n_cases = int(row["n_cases"])
            if n_cases != 20:
                raise ValueError(
                    f"Expected 20 cases for {method} at {level}, found {n_cases}"
                )
            rows[(level, method)] = float(row["throughput_gbps_mean"])

    levels = np.asarray(sorted({level for level, _ in rows}), dtype=float)
    if len(levels) != 17 or not np.allclose(
        levels, np.round(np.arange(0.8, 4.0 + 0.1, 0.2), 1)
    ):
        raise ValueError("LCT summary must contain the 0.8 to 4.0 sweep")

    throughput = np.asarray(
        [[rows[(level, method)] for method in METHOD_NAMES] for level in levels],
        dtype=float,
    )
    return levels, throughput


def load_for_results(results_path, n_cases=2):
    data = np.genfromtxt(results_path, delimiter=",")
    data = -data[:, [4, 8, 7, 6, 9]]
    return data.reshape(7, n_cases, len(METHOD_NAMES)).mean(axis=1)


def plot_method_lines(axis, x_values, data):
    lines = []
    for method_index, method_name in enumerate(METHOD_NAMES):
        style = METHOD_STYLES[method_index]
        (line,) = axis.plot(
            x_values,
            data[:, method_index],
            label=method_name,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.2,
            markersize=3.7,
            zorder=3,
        )
        lines.append(line)
    return lines


def make_figure(lct_summary_path, for_results_path):
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(FIG_WIDTH_IN, FIG_HEIGHT_IN)

    lct_values, lct_data = load_lct_summary(lct_summary_path)
    left_axis = axes[0]
    lines = plot_method_lines(left_axis, lct_values, lct_data)
    left_axis.set_position([0.125, 0.18, 0.39, 0.70])
    left_axis.set_ylabel("Network Throughput (Gbps)")
    left_axis.set_xlabel("Average Number of LCTs per Sat.")
    left_axis.set_xlim(0.72, 4.08)
    left_axis.set_xticks([0.8, 1.6, 2.4, 3.2, 4.0])
    left_axis.set_xticklabels(["0.8", "1.6", "2.4", "3.2", "4.0"])

    for_values = np.asarray([30, 40, 50, 60, 70, 80, 90])
    for_data = load_for_results(for_results_path)
    right_axis = axes[1]
    plot_method_lines(right_axis, for_values, for_data)
    right_axis.set_position([0.60, 0.18, 0.39, 0.70])
    right_axis.set_xlabel("Field of Regard Size (Degree)")
    right_axis.set_xticks(for_values)

    for axis in axes:
        axis.set_ylim(0, 360)
        axis.set_yticks([0, 100, 200, 300])
        axis.grid(True, zorder=0, alpha=0.35, linewidth=0.5)
    left_axis.set_yticklabels(["0", "100", "200", "300"])
    right_axis.tick_params(axis="y", labelleft=False)

    legend = fig.legend(
        lines,
        METHOD_NAMES,
        fontsize=FONT_SIZE - 1,
        loc="lower left",
        bbox_to_anchor=(0.125, 0.90, 0.865, 0.125),
        mode="expand",
        ncol=5,
        borderaxespad=0.0,
        handlelength=1.0,
        handleheight=0.8,
        handletextpad=0.1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
        borderpad=0.3,
        labelspacing=0.2,
    )
    legend.get_frame().set_linewidth(0.8)
    return fig


def main():
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parents[1]
    lct_summary_path = os.environ.get(
        "DEEP_LADU_LCT_SUMMARY",
        str(
            repo_root
            / "tmp"
            / "reviewer_lct_20260724"
            / "sweep-full"
            / "summary.csv"
        ),
    )
    for_results_path = os.environ.get(
        "DEEP_LADU_FOR_RESULTS",
        "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/"
        "test_ld_starlink_1000_varying_for/"
        "test_ld_starlink_1000_varying_for-2025-November-27-13-36-06-ail/ldl",
    )
    fig = make_figure(lct_summary_path, for_results_path)
    output_path = current_dir / f"{Path(__file__).stem}.pdf"
    print(f"Saving figure to {output_path}")
    fig.savefig(output_path, format="pdf", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
