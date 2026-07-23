import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPT = (
    REPO_ROOT
    / "sim_alg_j1_res"
    / "figure"
    / "plot_test_ld_starlink_1000_constellation_varying_n_sat_vs_sg.py"
)


def test_x_tick_labels_are_constellation_sizes():
    tree = ast.parse(PLOT_SCRIPT.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_xticklabels"
    ]

    assert calls, "the bar coordinates are being displayed as satellite counts"
    assert ast.literal_eval(calls[-1].args[0]) == [
        "500",
        "750",
        "1000",
        "1250",
        "1500",
    ]
