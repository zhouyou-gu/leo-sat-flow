import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


FONT_SIZE = 9
FIG_WIDTH_IN = 4.0
FIG_HEIGHT_IN = 2.5
LINE_WIDTH = 1.5
LEGEND_HANDLE_LENGTH = 0.8

SATELLITE_COUNTS = (500, 750, 1000, 1250, 1500)
SATELLITE_MARKERS = ("x", "+", "o", "s", r"$\ast$")
SATELLITE_MARKER_FILLSTYLES = ("full", "full", "none", "none", "full")
SATELLITE_MARKER_SIZES = (49, 64, 42, 42, 42)
SATELLITE_LEGEND_MARKER_SIZES = (6, 7, 5.75, 5.75, 5.75)
METHOD_LABELS = ("DeepLaDu", "MRate", "LaDu-20", "LaDu-100", "LaDu-200")
METHOD_COLORS = ("#1f77b4", "#d62728", "#D6B7FF", "#B378FF", "#6A00FF")

DEFAULT_RESULTS_DIR = Path(
    "/home/zhouyou/leo-sat-flow/sim_alg_j1_res/"
    "test_ld_starlink_1000_constellation_varying/"
    "test_ld_starlink_1000_constellation_varying-2025-October-27-12-56-15-ail"
)


def configure_plot_style():
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


def load_method_series(results_dir):
    results_dir = Path(results_dir)
    sg_data = np.genfromtxt(results_dir / "sg", delimiter=",")
    heu_data = np.genfromtxt(results_dir / "heu", delimiter=",")
    lml_data = np.genfromtxt(results_dir / "ldl", delimiter=",")

    method_times = np.empty((len(SATELLITE_COUNTS), len(METHOD_LABELS)))
    method_throughputs = np.empty_like(method_times)
    selected_iterations = np.array([20, 100, 300]) - 1

    for index, _ in enumerate(SATELLITE_COUNTS):
        sg_block = sg_data[index * 5000 : (index + 1) * 5000]
        sg_throughputs = -sg_block[:, 3].reshape(-1, 500).mean(axis=0)
        sg_times = sg_block[:, 2].reshape(-1, 500).mean(axis=0) / 1e6

        lml_block = lml_data[index * 10 : (index + 1) * 10]
        lml_throughput = -lml_block[:, 3].reshape(-1, 10).mean(axis=1)
        lml_time = lml_block[:, 2].reshape(-1, 10).mean(axis=1) / 1e6

        heu_block = heu_data[index * 10 : (index + 1) * 10]
        heu_throughput = -heu_block[:, 3].reshape(-1, 10).mean(axis=1)
        heu_time = heu_block[:, 2].reshape(-1, 10).mean(axis=1) / 1e6

        method_throughputs[index] = np.concatenate(
            (
                lml_throughput,
                heu_throughput,
                sg_throughputs[selected_iterations],
            )
        )
        method_times[index] = np.concatenate(
            (
                lml_time,
                heu_time,
                sg_times[selected_iterations],
            )
        )

    return method_times, method_throughputs


def create_figure(method_times, method_throughputs):
    configure_plot_style()
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(FIG_WIDTH_IN, FIG_HEIGHT_IN)

    method_handles = []
    for method_index, (label, color) in enumerate(
        zip(METHOD_LABELS, METHOD_COLORS)
    ):
        line, = ax.plot(
            method_times[:, method_index],
            method_throughputs[:, method_index],
            color=color,
            linestyle="-",
            linewidth=LINE_WIDTH,
            zorder=3,
        )
        line.set_gid(f"method:{label}")
        method_handles.append(
            Line2D([0], [0], color=color, linestyle="-", linewidth=LINE_WIDTH)
        )

    satellite_handles = []
    for satellite_index, (
        count,
        marker,
        marker_fillstyle,
        marker_size,
        legend_marker_size,
    ) in enumerate(
        zip(
            SATELLITE_COUNTS,
            SATELLITE_MARKERS,
            SATELLITE_MARKER_FILLSTYLES,
            SATELLITE_MARKER_SIZES,
            SATELLITE_LEGEND_MARKER_SIZES,
        )
    ):
        marker_colors = (
            {"facecolors": "none", "edgecolors": METHOD_COLORS}
            if marker_fillstyle == "none"
            else {"color": METHOD_COLORS}
        )
        points = ax.scatter(
            method_times[satellite_index],
            method_throughputs[satellite_index],
            marker=marker,
            s=marker_size,
            linewidths=1.5,
            zorder=5,
            **marker_colors,
        )
        points.set_gid(f"satellite:{count}")
        satellite_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="None",
                marker=marker,
                markersize=legend_marker_size,
                markerfacecolor=(
                    "none" if marker_fillstyle == "none" else "black"
                ),
                markeredgecolor="black",
                markeredgewidth=1.5,
            )
        )

    ax.set_position([0.125, 0.175, 0.825, 0.65])
    ax.set_xlabel(r"Computation Time (s)")
    ax.set_ylabel(r"Network Throughput (Gbps)")
    ax.set_xlim(0.001, 10000)
    ax.set_xscale("log")
    ax.set_ylim(30, 180)
    ax.grid(True, zorder=0)
    ax.axvline(x=1, color="black", linestyle="-", linewidth=2.5, zorder=1)
    ax.text(
        1.2,
        35,
        r"     Coherent time $\approx$ 1 second ",
        rotation=90,
        verticalalignment="bottom",
        color="black",
        fontsize=FONT_SIZE,
    )

    handles = method_handles + satellite_handles
    labels = list(METHOD_LABELS) + [
        rf"$I={count}$" for count in SATELLITE_COUNTS
    ]
    n_columns = 5
    n_rows = -(-len(handles) // n_columns)
    indices = np.arange(len(handles))
    padding = n_rows * n_columns - len(indices)
    indices = np.concatenate([indices, np.full(padding, -1)])
    order = indices.reshape(n_rows, n_columns).T.ravel()
    order = order[order >= 0]
    handles = [handles[index] for index in order]
    labels = [labels[index] for index in order]

    legend = fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.125, 0.835, 0.825, 0.1),
        mode="expand",
        ncol=n_columns,
        borderaxespad=0.0,
        handlelength=LEGEND_HANDLE_LENGTH,
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
    fig.add_artist(legend)

    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot connected method trajectories for the time-varying constellation."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the sg, heu, and ldl result files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".pdf"),
        help="Output PDF path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    method_times, method_throughputs = load_method_series(args.results_dir)
    fig, _ = create_figure(method_times, method_throughputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", pad_inches=0)
    plt.close(fig)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
