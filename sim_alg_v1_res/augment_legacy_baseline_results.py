#!/usr/bin/env python3

import io
import json
import math
import os
import sys
from collections import OrderedDict
from contextlib import redirect_stdout
from functools import lru_cache

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sim_mld.simulation import Simulation
from sim_mld.solver import mr_solver
from sim_mld.tle import generate_tle_partly_regular_constellation1000, load_url_tle_data
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT, STATS_OBJECT
from tcom_revision_common import DrLSnapshotSolver, current_paper_drl_variant, resolve_drl_model_path
from working_dir_path import get_working_dir_path


T0_UTC = (2025, 7, 16, 16, 0, 0)
DEFAULT_REMOTE_OLD_RESULTS_DIR = os.path.abspath(
    os.path.join(ROOT_DIR, os.pardir, "remote-old-results")
)
LEGACY_SOURCES = OrderedDict(
    [
        (
            "mixed_constellation",
            {
                "legacy_csv": os.path.join(
                    DEFAULT_REMOTE_OLD_RESULTS_DIR,
                    "test_dual_optimization_mixed_constellation",
                    "test_dual_optimization_mixed_constellation-2025-July-22-22-44-37-ail",
                    "res.csv",
                ),
                "traffic_seed": 500,
            },
        ),
        (
            "varying_for",
            {
                "legacy_csv": os.path.join(
                    DEFAULT_REMOTE_OLD_RESULTS_DIR,
                    "test_dual_optimization_starlink_1000_varying_for",
                    "test_dual_optimization_starlink_1000_varying_for-2025-July-31-20-51-49-ail",
                    "res.csv",
                ),
                "traffic_seed": 500,
            },
        ),
        (
            "lct_failure_rate",
            {
                "legacy_csv": os.path.join(
                    DEFAULT_REMOTE_OLD_RESULTS_DIR,
                    "test_dual_optimization_mixed_constellation_lct_failure_rate",
                    "test_dual_optimization_mixed_constellation_lct_failure_rate-2025-July-23-20-18-16-ail",
                    "res.csv",
                ),
                "traffic_seed": 500,
            },
        ),
        (
            "different_constellation",
            {
                "legacy_csv": os.path.join(
                    DEFAULT_REMOTE_OLD_RESULTS_DIR,
                    "test_dual_optimization_different_constellation",
                    "test_dual_optimization_different_constellation-2025-July-25-21-52-32-ail",
                    "res.csv",
                ),
                "traffic_seed": 1000,
            },
        ),
    ]
)

MERGED_COLUMN_ORDER = [
    "g_step",
    "iteration",
    "x_value",
    "y_value",
    "DuJo",
    "DRL",
    "+Grid",
    "MRate",
    "Rand",
    "SaTE",
]


class LegacySimulation(Simulation):
    def get_simulation_time(self):
        return self.ts.utc(*T0_UTC)


class LegacyFailureSimulation(LegacySimulation):
    LCT_FAILURE_RATE = 0.0

    def config_l_mask(self, seed=0):
        rng = np.random.default_rng(seed)
        self.lct_mask = np.zeros((self.n_sat, self.N_LCT_PER_SAT), dtype=np.float32)
        self.lct_mask[:, 0] = rng.choice(
            [0, 1],
            p=[self.LCT_FAILURE_RATE, 1 - self.LCT_FAILURE_RATE],
            size=self.n_sat,
        )
        self.lct_mask[:, 1] = rng.choice(
            [0, 1],
            p=[self.LCT_FAILURE_RATE, 1 - self.LCT_FAILURE_RATE],
            size=self.n_sat,
        )


def silence_debug_output():
    STATS_OBJECT._printalltime = lambda self, *args, **kwargs: None
    STATS_OBJECT._print = lambda self, *args, **kwargs: None


def run_quietly(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def ensure_2d(data):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_legacy_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing legacy results file {path}")
    return ensure_2d(np.genfromtxt(path, delimiter=","))


def legacy_file_root():
    base = get_working_dir_path()
    return {
        "starlink": os.path.join(base, "starlink_16_jul_2025_1600.tle"),
        "oneweb": os.path.join(base, "oneweb_16_jul_2025_1600.tle"),
    }


@lru_cache(maxsize=None)
def cached_starlink_constellation(n_sat, ratio):
    tle_paths = legacy_file_root()
    ts, valid_satellites, sat_array = run_quietly(
        generate_tle_partly_regular_constellation1000,
        n_sat=int(n_sat),
        ratio=float(ratio),
        starlink_tle_path=tle_paths["starlink"],
        seed=0,
    )
    return ts, sat_array


@lru_cache(maxsize=None)
def cached_oneweb_constellation():
    tle_paths = legacy_file_root()
    ts, valid_satellites, sat_array = run_quietly(
        load_url_tle_data,
        tle_paths["oneweb"],
        True,
    )
    return ts, sat_array


def build_mixed_constellation_simulation(n_sat):
    ts, sat_array = cached_starlink_constellation(int(n_sat), 0.0)
    simulation = LegacySimulation(ts, sat_array)
    simulation.config_l_mask(seed=0)
    simulation.update_space()
    return simulation


def build_varying_for_simulation(for_value):
    ts, sat_array = cached_starlink_constellation(1000, 0.0)
    simulation = LegacySimulation(ts, sat_array)
    simulation.FOR_THETA_HALF = float(for_value)
    simulation.cos_threshold = math.cos(math.radians(simulation.FOR_THETA_HALF))
    simulation.config_l_mask(seed=0)
    simulation.update_space()
    return simulation


def build_lct_failure_simulation(rho):
    ts, sat_array = cached_starlink_constellation(1000, 0.0)
    simulation = LegacyFailureSimulation(ts, sat_array)
    simulation.LCT_FAILURE_RATE = float(rho)
    simulation.config_l_mask(seed=0)
    simulation.update_space()
    return simulation


def build_different_constellation_simulation(constellation_name):
    if constellation_name == "oneweb":
        ts, sat_array = cached_oneweb_constellation()
    elif constellation_name == "starlink":
        ts, sat_array = cached_starlink_constellation(1000, 0.0)
    elif constellation_name == "walker-delta":
        ts, sat_array = cached_starlink_constellation(1000, 1.0)
    else:
        raise ValueError(f"Unsupported constellation {constellation_name!r}")

    simulation = LegacySimulation(ts, sat_array)
    if constellation_name == "oneweb":
        simulation.terrain.GW_RANGE = 1800.0
        simulation.LISL_MAX_DISTANCE = 4000.0
    simulation.config_l_mask(seed=0)
    simulation.update_space()
    return simulation


def evaluate_drl(build_simulation, traffic_seed):
    simulation = run_quietly(build_simulation)
    solver = DrLSnapshotSolver()
    simulation.set_solver(solver)
    simulation.update_solver_traffic_info(seed=int(traffic_seed))
    run_quietly(
        solver.load_drl,
        path=resolve_drl_model_path(variant=current_paper_drl_variant()),
        variant=current_paper_drl_variant(),
    )
    run_quietly(solver.infer_drl)
    p_o, rates, srouting, aux, connected_sat, connected_lct = run_quietly(
        solver.get_prim_objective,
        with_rates=True,
    )
    return float(p_o)


def evaluate_sate(build_simulation, traffic_seed):
    simulation = run_quietly(build_simulation)
    solver = mr_solver()
    simulation.set_solver(solver)
    simulation.update_solver_traffic_info(seed=int(traffic_seed))
    p_o, rates, srouting, aux, connected_sat, connected_lct = run_quietly(
        solver.get_prim_objective_heuristic,
        with_rates=True,
        matching_method="mwm",
        routing_method="spf",
        seed=0,
    )
    return float(p_o)


def merge_rows(old_rows, deep_ladu_values, sate_values):
    merged_rows = []
    for old_row, deep_value, sate_value in zip(old_rows, deep_ladu_values, sate_values):
        merged_rows.append(
            [
                old_row[0],
                old_row[1],
                old_row[2],
                old_row[3],
                old_row[4],
                deep_value,
                old_row[5],
                old_row[6],
                old_row[7],
                sate_value,
            ]
        )
    return np.asarray(merged_rows, dtype=np.float64)


def write_csv(path, data):
    np.savetxt(path, data, delimiter=",")


def main():
    silence_debug_output()
    output_dir = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "remote_old_results_dir": DEFAULT_REMOTE_OLD_RESULTS_DIR,
        "merged_column_order": MERGED_COLUMN_ORDER,
        "drl_variant": current_paper_drl_variant(),
        "drl_model_path": os.getenv("TCOM_REVISION_DRL_MODEL_PATH", "").strip() or None,
        "experiments": {},
    }

    mixed_old_rows = load_legacy_csv(LEGACY_SOURCES["mixed_constellation"]["legacy_csv"])
    mixed_drl = []
    mixed_sate = []
    for row in mixed_old_rows:
        n_sat = int(row[2])
        build_sim = lambda n_sat=n_sat: build_mixed_constellation_simulation(n_sat)
        mixed_drl.append(
            evaluate_drl(build_sim, LEGACY_SOURCES["mixed_constellation"]["traffic_seed"])
        )
        mixed_sate.append(
            evaluate_sate(build_sim, LEGACY_SOURCES["mixed_constellation"]["traffic_seed"])
        )
    mixed_merged = merge_rows(mixed_old_rows, mixed_drl, mixed_sate)
    mixed_path = os.path.join(output_dir, "mixed_constellation_merged.csv")
    write_csv(mixed_path, mixed_merged)
    manifest["experiments"]["mixed_constellation"] = {
        "legacy_csv": LEGACY_SOURCES["mixed_constellation"]["legacy_csv"],
        "merged_csv": mixed_path,
        "row_count": int(mixed_merged.shape[0]),
    }

    varying_for_old_rows = load_legacy_csv(LEGACY_SOURCES["varying_for"]["legacy_csv"])
    varying_for_drl = []
    varying_for_sate = []
    for row in varying_for_old_rows:
        for_value = float(row[3])
        build_sim = lambda for_value=for_value: build_varying_for_simulation(for_value)
        varying_for_drl.append(
            evaluate_drl(build_sim, LEGACY_SOURCES["varying_for"]["traffic_seed"])
        )
        varying_for_sate.append(
            evaluate_sate(build_sim, LEGACY_SOURCES["varying_for"]["traffic_seed"])
        )
    varying_for_merged = merge_rows(varying_for_old_rows, varying_for_drl, varying_for_sate)
    varying_for_path = os.path.join(output_dir, "varying_for_merged.csv")
    write_csv(varying_for_path, varying_for_merged)
    manifest["experiments"]["varying_for"] = {
        "legacy_csv": LEGACY_SOURCES["varying_for"]["legacy_csv"],
        "merged_csv": varying_for_path,
        "row_count": int(varying_for_merged.shape[0]),
    }

    lct_failure_old_rows = load_legacy_csv(LEGACY_SOURCES["lct_failure_rate"]["legacy_csv"])
    lct_failure_drl = []
    lct_failure_sate = []
    for row in lct_failure_old_rows:
        rho = float(row[3])
        build_sim = lambda rho=rho: build_lct_failure_simulation(rho)
        lct_failure_drl.append(
            evaluate_drl(build_sim, LEGACY_SOURCES["lct_failure_rate"]["traffic_seed"])
        )
        lct_failure_sate.append(
            evaluate_sate(build_sim, LEGACY_SOURCES["lct_failure_rate"]["traffic_seed"])
        )
    lct_failure_merged = merge_rows(lct_failure_old_rows, lct_failure_drl, lct_failure_sate)
    lct_failure_path = os.path.join(output_dir, "lct_failure_rate_merged.csv")
    write_csv(lct_failure_path, lct_failure_merged)
    manifest["experiments"]["lct_failure_rate"] = {
        "legacy_csv": LEGACY_SOURCES["lct_failure_rate"]["legacy_csv"],
        "merged_csv": lct_failure_path,
        "row_count": int(lct_failure_merged.shape[0]),
    }

    different_constellation_old_rows = load_legacy_csv(
        LEGACY_SOURCES["different_constellation"]["legacy_csv"]
    )
    constellation_order = ["oneweb", "starlink", "walker-delta"]
    different_constellation_drl = []
    different_constellation_sate = []
    for constellation_name in constellation_order:
        build_sim = lambda constellation_name=constellation_name: build_different_constellation_simulation(
            constellation_name
        )
        different_constellation_drl.append(
            evaluate_drl(build_sim, LEGACY_SOURCES["different_constellation"]["traffic_seed"])
        )
        different_constellation_sate.append(
            evaluate_sate(build_sim, LEGACY_SOURCES["different_constellation"]["traffic_seed"])
        )
    different_constellation_merged = merge_rows(
        different_constellation_old_rows,
        different_constellation_drl,
        different_constellation_sate,
    )
    different_constellation_path = os.path.join(output_dir, "different_constellation_merged.csv")
    write_csv(different_constellation_path, different_constellation_merged)
    manifest["experiments"]["different_constellation"] = {
        "legacy_csv": LEGACY_SOURCES["different_constellation"]["legacy_csv"],
        "merged_csv": different_constellation_path,
        "row_count": int(different_constellation_merged.shape[0]),
        "legacy_row_order": constellation_order,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)


if __name__ == "__main__":
    main()
