import os

import matplotlib.pyplot as plt


FONT_SIZE = 9
METHOD_NAMES = ["DuJo", "DRL", "SaTE", "MRate", "+Grid", "Rand"]
METHOD_COLUMN_INDICES = [4, 5, 9, 7, 6, 8]
METHOD_COLORS = {
    "DuJo": "#1f77b4",
    "DRL": "#ff7f0e",
    "DeepLaDu": "#ff7f0e",
    "SaTE": "#2ca02c",
    "MRate": "#d62728",
    "+Grid": "#9467bd",
    "Rand": "#8c564b",
}


def latest_augmentation_file(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_dir = os.path.join(base_dir, "augment_legacy_baseline_results")
    candidate_paths = [
        os.path.join(parent_dir, entry, filename)
        for entry in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, entry))
        and os.path.exists(os.path.join(parent_dir, entry, filename))
    ]
    if not candidate_paths:
        raise FileNotFoundError(f"No augmentation run directories found under {parent_dir}")
    return max(candidate_paths, key=os.path.getmtime)


def apply_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )
    plt.rc("mathtext", fontset="cm")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
