import hashlib
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SCRIPT = (
    REPO_ROOT
    / "sim_alg_j1_res"
    / "figure"
    / "plot_test_ld_starlink_1000_constellation_thr_tim_single.py"
)
NEW_SCRIPT = (
    REPO_ROOT
    / "sim_alg_j1_res"
    / "figure"
    / "plot_test_ld_starlink_1000_constellation_thr_tim_connected.py"
)
ORIGINAL_SHA256 = "a572cb9a452e4387f272987b126164089dc7d91e976a16ee206b8bf5e8aa5612"


def load_plot_module():
    assert NEW_SCRIPT.exists(), "the new connected-marker plotter has not been created"
    spec = importlib.util.spec_from_file_location("connected_plot", NEW_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_plot_script_is_unchanged():
    digest = hashlib.sha256(ORIGINAL_SCRIPT.read_bytes()).hexdigest()
    assert digest == ORIGINAL_SHA256


def test_new_plot_connects_each_method_and_uses_line_markers():
    module = load_plot_module()

    assert module.SATELLITE_MARKERS == ("x", "+", "o", "s", r"$\ast$")
    assert module.SATELLITE_MARKER_FILLSTYLES == (
        "full",
        "full",
        "none",
        "none",
        "full",
    )
    assert module.SATELLITE_MARKER_SIZES == (49, 64, 42, 42, 42)
    assert module.SATELLITE_LEGEND_MARKER_SIZES == (6, 7, 5.75, 5.75, 5.75)
    assert getattr(module, "LEGEND_HANDLE_LENGTH", None) == 0.8

    method_times = np.array(
        [
            [0.01, 0.02, 10.0, 20.0, 30.0],
            [0.02, 0.03, 20.0, 30.0, 40.0],
            [0.03, 0.04, 30.0, 40.0, 50.0],
            [0.04, 0.05, 40.0, 50.0, 60.0],
            [0.05, 0.06, 50.0, 60.0, 70.0],
        ]
    )
    method_throughputs = np.array(
        [
            [50.0, 45.0, 40.0, 35.0, 30.0],
            [70.0, 65.0, 60.0, 55.0, 50.0],
            [90.0, 85.0, 80.0, 75.0, 70.0],
            [110.0, 105.0, 100.0, 95.0, 90.0],
            [130.0, 125.0, 120.0, 115.0, 110.0],
        ]
    )

    fig, ax = module.create_figure(method_times, method_throughputs)
    method_lines = [
        line for line in ax.lines if (line.get_gid() or "").startswith("method:")
    ]

    assert len(method_lines) == len(module.METHOD_LABELS)
    assert all(len(line.get_xdata()) == len(module.SATELLITE_COUNTS) for line in method_lines)
    satellite_collections = [
        collection
        for collection in ax.collections
        if (collection.get_gid() or "").startswith("satellite:")
    ]
    assert len(satellite_collections) == len(module.SATELLITE_COUNTS)
    assert [
        collection.get_sizes()[0] for collection in satellite_collections
    ] == list(module.SATELLITE_MARKER_SIZES)
    assert [
        collection.get_facecolors().size for collection in satellite_collections
    ] == [4 * len(module.METHOD_COLORS), 4 * len(module.METHOD_COLORS), 0, 0, 4 * len(module.METHOD_COLORS)]

    plt.close(fig)
