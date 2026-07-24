#!/usr/bin/env python3
"""Plot congestion-price CDFs for exact two-, three-, and four-LCT tests."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FONT_SIZE = 9
FIG_WIDTH_IN = 3.5
FIG_HEIGHT_IN = 2.25
LCT_STYLES = {
    2: {
        "label": "2 LCTs",
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
        "markevery": (0.00, 0.12),
    },
    3: {
        "label": "3 LCTs",
        "color": "#D55E00",
        "linestyle": "--",
        "marker": "s",
        "markevery": (0.04, 0.12),
    },
    4: {
        "label": "4 LCTs",
        "color": "#009E73",
        "linestyle": "-.",
        "marker": "^",
        "markevery": (0.08, 0.12),
    },
}


def configure_matplotlib():
    """Apply the typography used by the existing manuscript figures."""
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


def empirical_cdf(values):
    """Return sorted observations and their empirical cumulative probabilities."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("CDF values must be a non-empty one-dimensional array")
    ordered = np.sort(values)
    probabilities = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(
        ordered
    )
    return ordered, probabilities


def load_lct_count_prices(samples_path):
    """Pool selected-endpoint prices within each exact LCT-count case."""
    grouped = {lct_count: [] for lct_count in LCT_STYLES}
    with Path(samples_path).open(newline="", encoding="utf-8") as samples_file:
        for row in csv.DictReader(samples_file):
            lct_count = int(row["lct_count"])
            if lct_count not in grouped:
                raise ValueError(f"Unknown exact LCT count {lct_count}")
            grouped[lct_count].append(float(row["price"]))
    if any(not values for values in grouped.values()):
        raise ValueError("Price samples are required for 2, 3, and 4 LCTs")
    return {
        lct_count: np.asarray(values, dtype=np.float64)
        for lct_count, values in grouped.items()
    }


def make_figure(samples_path):
    """Create the three-line empirical CDF figure."""
    configure_matplotlib()
    lct_count_prices = load_lct_count_prices(samples_path)
    figure, axis = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    for lct_count, style in LCT_STYLES.items():
        values, probabilities = empirical_cdf(lct_count_prices[lct_count])
        axis.step(
            values,
            probabilities,
            where="post",
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.2,
            marker=style["marker"],
            markevery=style["markevery"],
            markersize=4.2,
            markerfacecolor="none",
            markeredgewidth=0.8,
            zorder=3,
        )

    maximum_price = max(np.max(values) for values in lct_count_prices.values())
    axis.set_xlim(0.0, 1.02 * maximum_price)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Congestion Price")
    axis.set_ylabel("Empirical CDF")
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.grid(True, alpha=0.35, linewidth=0.5)
    legend = axis.legend(
        loc="lower right",
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        borderpad=0.3,
        handlelength=1.8,
        handletextpad=0.4,
    )
    legend.get_frame().set_linewidth(0.8)
    figure.tight_layout(pad=0.35)
    return figure


def parse_args():
    """Parse command-line paths for the CDF input and output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main():
    """Render the CDF figure to the requested output path."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(args.samples)
    figure.savefig(args.output, pad_inches=0)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
