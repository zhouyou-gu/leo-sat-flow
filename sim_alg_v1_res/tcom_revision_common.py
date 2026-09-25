#!/usr/bin/env python3

import glob
import json
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import scipy.sparse as sp

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sim_mld.simulation import Simulation
from sim_mld.solver import *
from sim_mld.tle import generate_tle_starlink_shell_block_constellation
from sim_mld.ml.e2e_rl.model import rl_model


T0_UTC = datetime(2025, 7, 16, 16, 0, 0)
DEFAULT_TIME_OFFSETS_MINUTES = [0, 30, 60, 90, 120, 180]
DEFAULT_ATP_SNAPSHOT_INTERVAL_SEC = 10
DEFAULT_ATP_TIME_SEC = 50
DEFAULT_ATP_TIME_SWEEP_SEC = [0, 10, 20, 30, 50, 70, 90]
DEFAULT_ATP_EVALUATION_SECONDS = 6 * 60
DEFAULT_ACTIVE_USER_PERCENTAGES = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
DEFAULT_DUJO_STEPS = 500
DEFAULT_TRAFFIC_SEED = 0
DEFAULT_DUJO_TRAFFIC_MODE = "legacy"
DEFAULT_DUJO_EVAL_MODE = "last"
DEFAULT_DRL_VARIANT = "pg"
DEFAULT_DRL_BETA = 0.7
DEFAULT_DRL_GAMMA = 1.0

HEADLINE_METHODS = ["DuJo", "DRL", "SaTE", "MRate", "+Grid"]
SUPPORTING_METHODS = ["Rand"]

HEURISTIC_METHOD_SPECS = {
    "SaTE": {"matching_method": "mwm", "routing_method": "spf"},
    "MRate": {"matching_method": "mwm", "routing_method": "ospf"},
    "+Grid": {"matching_method": "grid", "routing_method": "ospf"},
    "Rand": {"matching_method": "rand", "routing_method": "ospf"},
}


def rand_case_seed(offset_minutes, active_user_percentage, traffic_seed):
    active_user_key = int(round(float(active_user_percentage) * 1e8))
    return int(traffic_seed) + 1009 * int(offset_minutes) + 9173 * active_user_key


def rand_case_seed_seconds(offset_seconds, active_user_percentage, traffic_seed):
    active_user_key = int(round(float(active_user_percentage) * 1e8))
    return int(traffic_seed) + 1009 * int(offset_seconds) + 9173 * active_user_key


def parse_env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def parse_env_choice(name, default, valid_choices):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    value = value.strip().lower()
    if value not in valid_choices:
        raise ValueError(f"{name} must be one of {sorted(valid_choices)}, got {value!r}")
    return value


def parse_env_float_list(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_env_int_list(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def current_paper_drl_variant():
    return parse_env_choice(
        "TCOM_REVISION_DRL_VARIANT",
        DEFAULT_DRL_VARIANT,
        {"dpg", "pg"},
    )


def drl_model_search_patterns(variant):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    selected_nn_dir = os.path.join(base_dir, "sim_alg_j1_res", "selected_nn")
    variant_prefix = variant.lower()
    return [
        os.path.join(selected_nn_dir, f"{variant_prefix}_model.model_final*.pt"),
        os.path.join(selected_nn_dir, f"{variant_prefix}_gnnsolver.model_final*.pt"),
        os.path.join(selected_nn_dir, f"rl_model.model_final*.pt"),
        os.path.join(
            base_dir,
            "sim_alg_j1_res",
            f"train_rl_starlink_1000_{variant_prefix}",
            "**",
            "*.pt",
        ),
    ]


def resolve_drl_model_path(model_path=None, variant=None):
    if variant is None:
        variant = current_paper_drl_variant()
    if model_path is None:
        model_path = os.getenv("TCOM_REVISION_DRL_MODEL_PATH", "").strip() or None
    if model_path is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Configured DRL model path does not exist: {model_path}")
        return model_path

    candidates = []
    for pattern in drl_model_search_patterns(variant):
        candidates.extend(glob.glob(pattern, recursive=True))
    candidates = [path for path in candidates if os.path.isfile(path)]
    if candidates:
        preferred = [path for path in candidates if "_target." not in os.path.basename(path)]
        if preferred:
            return max(preferred, key=os.path.getmtime)
        return max(candidates, key=os.path.getmtime)

    raise FileNotFoundError(
        "No trained DRL checkpoint was found for the current-paper benchmark. "
        "Set TCOM_REVISION_DRL_MODEL_PATH to a saved rl_model checkpoint."
    )


def default_starlink_tle_path():
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.path.pardir,
        "starlink_16_jul_2025_1600.tle",
    )


def snapshot_datetime(offset_minutes):
    return T0_UTC + timedelta(minutes=int(offset_minutes))


def snapshot_datetime_seconds(offset_seconds):
    return T0_UTC + timedelta(seconds=int(offset_seconds))


def snapshot_time_scale(ts, offset_minutes):
    dt = snapshot_datetime(offset_minutes)
    return ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def snapshot_time_scale_seconds(ts, offset_seconds):
    dt = snapshot_datetime_seconds(offset_seconds)
    return ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def snapshot_time_scale_from_offset(ts, offset_minutes, offset_seconds=None):
    if offset_seconds is None:
        return snapshot_time_scale(ts, offset_minutes)
    return snapshot_time_scale_seconds(ts, offset_seconds)


def latest_run_dir(parent_dir):
    candidates = [
        os.path.join(parent_dir, entry)
        for entry in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, entry))
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories were found under {parent_dir}")
    return max(candidates, key=os.path.getmtime)


def load_shell_constellation(n_sat=1000, starlink_tle_path=None):
    if starlink_tle_path is None:
        starlink_tle_path = default_starlink_tle_path()
    ts, valid_satellites, sat_array, metadata = generate_tle_starlink_shell_block_constellation(
        n_sat=n_sat,
        starlink_tle_path=starlink_tle_path,
    )
    metadata = dict(metadata)
    metadata["tle_path"] = starlink_tle_path
    metadata["t0_utc"] = T0_UTC.isoformat()
    metadata["selected_satellites"] = [sat.name for sat in valid_satellites]
    return ts, valid_satellites, sat_array, metadata


def save_shell_metadata(output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "shell_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


class RevisionLpdSolver(mr_solver):
    def update_step_rates_prices(self):
        self.N_STEP += 1
        step_size = self.ALPHA / (self.N_STEP ** self.BETA)
        connected_sat, _ = self.get_dual_matching()

        costs, lengths, paths_all = self.get_dual_srouting()
        srouting = construct_edges_matrix_all_in_one(
            self.data_source,
            self.data_target,
            costs,
            lengths,
            paths_all,
        )
        self.s_t_traffic_rates = self.get_rates_dual(costs=costs)

        srouting = construct_edges_matrix_all_in_one(
            self.data_source,
            self.data_target,
            self.s_t_traffic_rates,
            lengths,
            paths_all,
        )
        if np.asarray(srouting).size == 0:
            qx_csr = sp.csr_matrix((self.n_sat, self.n_sat))
        else:
            qx = build_csr(self.n_sat, srouting[:, 0:2], srouting[:, 4], merging_method="sum")
            qx_csr = sp.csr_matrix((qx[2], qx[1], qx[0]), shape=(self.n_sat, self.n_sat))

        if np.asarray(connected_sat).size == 0:
            rc_csr = sp.csr_matrix((self.n_sat, self.n_sat))
        else:
            edge_in_both_direction = np.concatenate((connected_sat, connected_sat[:, ::-1]), axis=0)
            capacity_matched = self.compute_capacity(
                np.linalg.norm(
                    self.positions[edge_in_both_direction[:, 0]]
                    - self.positions[edge_in_both_direction[:, 1]],
                    axis=1,
                )
            )
            rc = build_csr(
                self.n_sat,
                edge_in_both_direction,
                capacity_matched,
                merging_method="sum",
                sym_half=False,
            )
            rc_csr = sp.csr_matrix((rc[2], rc[1], rc[0]), shape=(self.n_sat, self.n_sat))

        dif_price_graph = step_size * (qx_csr - rc_csr)
        self.price_graph.add_prices(dif_price_graph)


class DrLSnapshotSolver(mr_solver):
    def load_drl(
        self,
        path,
        variant=None,
        beta=DEFAULT_DRL_BETA,
        gamma=DEFAULT_DRL_GAMMA,
    ):
        if variant is None:
            variant = current_paper_drl_variant()
        det = variant == "dpg"
        self.model = rl_model(BETA=beta, GAMMA=gamma, DET=det)
        self.model.load_model(path)
        self.model.eval()

    def infer_drl(self):
        capacity = self.compute_capacity(
            np.linalg.norm(
                self.positions[self.possible_sat_pair_expanded[:, 0]]
                - self.positions[self.possible_sat_pair_expanded[:, 1]],
                axis=1,
            )
        )
        indptr, indices, data = build_csr(
            self.n_sat,
            self.possible_sat_pair_expanded,
            capacity,
            merging_method="avg",
            sym_half=True,
        )
        cp_edge_list = csr_to_edge_list(indptr, indices, data)
        possible_sat_pair_non_expanded_sym = cp_edge_list[:, 0:2]
        possible_capacity_non_expanded_sym = cp_edge_list[:, 2]

        sat_capacity = self.forward_traffic_capacity
        sat_demand = self.forward_traffic_demand
        x = np.concatenate((sat_capacity.reshape(-1, 1), sat_demand.reshape(-1, 1)), axis=1)

        prices = self.model.get_output_np_edge_weight(
            x,
            possible_sat_pair_non_expanded_sym,
            possible_capacity_non_expanded_sym,
            use_target=False,
        )
        self.price_graph.price_graph = sp.csr_matrix(
            (
                prices,
                (
                    possible_sat_pair_non_expanded_sym[:, 0],
                    possible_sat_pair_non_expanded_sym[:, 1],
                ),
            ),
            shape=(self.n_sat, self.n_sat),
        )


class SnapshotDualSimulation(Simulation):
    def __init__(
        self,
        ts,
        sat_array,
        snapshot_time,
        traffic_seed=DEFAULT_TRAFFIC_SEED,
        dujo_traffic_mode=DEFAULT_DUJO_TRAFFIC_MODE,
        dujo_eval_mode=DEFAULT_DUJO_EVAL_MODE,
    ):
        self.snapshot_time = snapshot_time
        self.traffic_seed = traffic_seed
        self.last_result = None
        self.best_result = None
        self.dujo_traffic_mode = dujo_traffic_mode
        self.dujo_eval_mode = dujo_eval_mode
        self._loaded_traffic_seed = None
        super().__init__(ts, sat_array)
        self.current_time = snapshot_time

    def get_simulation_time(self):
        return self.snapshot_time

    def run_step(self):
        if self.filtered_lct_pair_expanded.size == 0:
            self.last_result = None
            return

        if self.dujo_traffic_mode == "legacy":
            traffic_seed = self.N_STEP
        elif self.dujo_traffic_mode == "fixed":
            traffic_seed = self.traffic_seed
        else:
            raise ValueError(f"Unsupported DuJo traffic mode {self.dujo_traffic_mode!r}")

        # In fixed mode, the traffic state is identical across DuJo iterations for
        # the same snapshot and seed, so reloading it every step is unnecessary.
        if self._loaded_traffic_seed != traffic_seed:
            self.update_solver_traffic_info(seed=traffic_seed)
            self._loaded_traffic_seed = traffic_seed
        self.solver.update_step_rates_prices()
        d_o, _ = self.solver.get_dual_objective(with_prices=True)
        p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = (
            self.solver.get_prim_objective(with_rates=True)
        )
        result = {
            "dual_objective": float(-d_o),
            "primal_objective": float(-p_o),
            "rates": rates,
            "costs": costs,
            "lengths": lengths,
            "paths_all": paths_all,
            "srouting": srouting,
            "connected_sat": connected_sat,
            "connected_lct": connected_lct,
        }
        self.last_result = result
        if (
            self.best_result is None
            or result["primal_objective"] > self.best_result["primal_objective"]
        ):
            self.best_result = result


def configure_simulation(simulation, solver, active_user_percentage):
    simulation.terrain.ACTIVE_USER_PERCENTAGE = active_user_percentage
    simulation.config_l_mask(seed=0)
    simulation.update_space()
    simulation.set_solver(solver)
    return simulation


def compute_metrics(simulation, method_name, rates, lengths, connected_sat):
    offered = float(np.sum(simulation.solver.forward_traffic_demand))
    served = float(np.sum(rates))
    unmet = max(offered - served, 0.0)
    served_ratio = 1.0 if offered <= 0 else served / offered
    unmet_fraction = 0.0 if offered <= 0 else unmet / offered
    positive_lengths = lengths[lengths > 0]
    connected_flows = int(np.count_nonzero(rates > 1e-9))
    return {
        "method": method_name,
        "offered_demand_gbps": offered,
        "served_throughput_gbps": served,
        "served_ratio": served_ratio,
        "unmet_demand_gbps": unmet,
        "unmet_fraction": unmet_fraction,
        "connected_flows": connected_flows,
        "avg_hops": float(np.mean(positive_lengths)) if positive_lengths.size else 0.0,
        "matched_satellite_pairs": int(connected_sat.shape[0]) if np.asarray(connected_sat).size else 0,
    }


def collect_dujo_result(simulation, traffic_seed):
    simulation.update_solver_traffic_info(seed=traffic_seed)
    d_o, _ = simulation.solver.get_dual_objective(with_prices=True)
    p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = (
        simulation.solver.get_prim_objective(with_rates=True)
    )
    result = {
        "dual_objective": float(-d_o),
        "primal_objective": float(-p_o),
        "rates": rates,
        "costs": costs,
        "lengths": lengths,
        "paths_all": paths_all,
        "srouting": srouting,
        "connected_sat": connected_sat,
        "connected_lct": connected_lct,
    }
    result.update(
        compute_metrics(
            simulation,
            "DuJo",
            rates,
            lengths,
            connected_sat,
        )
    )
    return result


def evaluate_dujo(
    ts,
    sat_array,
    offset_minutes,
    active_user_percentage,
    traffic_seed=DEFAULT_TRAFFIC_SEED,
    dujo_steps=DEFAULT_DUJO_STEPS,
    dujo_traffic_mode=DEFAULT_DUJO_TRAFFIC_MODE,
    dujo_eval_mode=DEFAULT_DUJO_EVAL_MODE,
    offset_seconds=None,
    include_simulation=False,
):
    solver = RevisionLpdSolver()
    simulation = SnapshotDualSimulation(
        ts,
        sat_array,
        snapshot_time_scale_from_offset(ts, offset_minutes, offset_seconds=offset_seconds),
        traffic_seed=traffic_seed,
        dujo_traffic_mode=dujo_traffic_mode,
        dujo_eval_mode=dujo_eval_mode,
    )
    configure_simulation(simulation, solver, active_user_percentage)
    simulation.run(TOT_STEPS=dujo_steps, visualize=False)
    if dujo_eval_mode == "best":
        raise NotImplementedError(
            "dujo_eval_mode='best' is not supported for the current paper-local benchmark "
            "because the final reported comparison is re-evaluated on a common traffic seed."
        )
    result = collect_dujo_result(simulation, traffic_seed=traffic_seed)
    if result is None:
        raise RuntimeError("DuJo produced no result because no feasible LISLs were found.")
    if include_simulation:
        result["_simulation"] = simulation
    return dict(result)


def evaluate_heuristic(
    method_name,
    ts,
    sat_array,
    offset_minutes,
    active_user_percentage,
    traffic_seed=DEFAULT_TRAFFIC_SEED,
    offset_seconds=None,
    include_simulation=False,
):
    spec = HEURISTIC_METHOD_SPECS[method_name]
    matching_seed = int(traffic_seed)
    if method_name == "Rand":
        if offset_seconds is None:
            matching_seed = rand_case_seed(offset_minutes, active_user_percentage, traffic_seed)
        else:
            matching_seed = rand_case_seed_seconds(offset_seconds, active_user_percentage, traffic_seed)
    solver = mr_solver()
    simulation = SnapshotDualSimulation(
        ts,
        sat_array,
        snapshot_time_scale_from_offset(ts, offset_minutes, offset_seconds=offset_seconds),
        traffic_seed=traffic_seed,
    )
    configure_simulation(simulation, solver, active_user_percentage)
    simulation.update_solver_traffic_info(seed=traffic_seed)
    p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = (
        simulation.solver.get_prim_objective_heuristic(
            with_rates=True,
            matching_method=spec["matching_method"],
            routing_method=spec["routing_method"],
            seed=matching_seed,
        )
    )
    result = {
        "dual_objective": math.nan,
        "primal_objective": float(-p_o),
        "rates": rates,
        "costs": costs,
        "lengths": lengths,
        "paths_all": paths_all,
        "srouting": srouting,
        "connected_sat": connected_sat,
        "connected_lct": connected_lct,
    }
    result.update(compute_metrics(simulation, method_name, rates, lengths, connected_sat))
    if include_simulation:
        result["_simulation"] = simulation
    return result


def evaluate_drl(
    ts,
    sat_array,
    offset_minutes,
    active_user_percentage,
    traffic_seed=DEFAULT_TRAFFIC_SEED,
    model_path=None,
    offset_seconds=None,
    include_simulation=False,
):
    variant = current_paper_drl_variant()
    model_path = resolve_drl_model_path(model_path=model_path, variant=variant)
    solver = DrLSnapshotSolver()
    simulation = SnapshotDualSimulation(
        ts,
        sat_array,
        snapshot_time_scale_from_offset(ts, offset_minutes, offset_seconds=offset_seconds),
        traffic_seed=traffic_seed,
    )
    configure_simulation(simulation, solver, active_user_percentage)
    simulation.update_solver_traffic_info(seed=traffic_seed)
    solver.load_drl(path=model_path, variant=variant)
    solver.infer_drl()
    d_o, _ = solver.get_dual_objective(with_prices=True)
    p_o, rates, srouting, (costs, lengths, paths_all), connected_sat, connected_lct = (
        solver.get_prim_objective(with_rates=True)
    )
    result = {
        "dual_objective": float(-d_o),
        "primal_objective": float(-p_o),
        "rates": rates,
        "costs": costs,
        "lengths": lengths,
        "paths_all": paths_all,
        "srouting": srouting,
        "connected_sat": connected_sat,
        "connected_lct": connected_lct,
    }
    result.update(compute_metrics(simulation, "DRL", rates, lengths, connected_sat))
    if include_simulation:
        result["_simulation"] = simulation
    return result


def evaluate_method(
    method_name,
    ts,
    sat_array,
    offset_minutes,
    active_user_percentage,
    traffic_seed=DEFAULT_TRAFFIC_SEED,
    dujo_steps=DEFAULT_DUJO_STEPS,
    dujo_traffic_mode=DEFAULT_DUJO_TRAFFIC_MODE,
    dujo_eval_mode=DEFAULT_DUJO_EVAL_MODE,
    offset_seconds=None,
    include_simulation=False,
):
    if method_name == "DuJo":
        return evaluate_dujo(
            ts,
            sat_array,
            offset_minutes,
            active_user_percentage,
            traffic_seed=traffic_seed,
            dujo_steps=dujo_steps,
            dujo_traffic_mode=dujo_traffic_mode,
            dujo_eval_mode=dujo_eval_mode,
            offset_seconds=offset_seconds,
            include_simulation=include_simulation,
        )
    if method_name == "DRL":
        return evaluate_drl(
            ts,
            sat_array,
            offset_minutes,
            active_user_percentage,
            traffic_seed=traffic_seed,
            offset_seconds=offset_seconds,
            include_simulation=include_simulation,
        )
    return evaluate_heuristic(
        method_name,
        ts,
        sat_array,
        offset_minutes,
        active_user_percentage,
        traffic_seed=traffic_seed,
        offset_seconds=offset_seconds,
        include_simulation=include_simulation,
    )


def evaluate_method_seconds(
    method_name,
    ts,
    sat_array,
    offset_seconds,
    active_user_percentage,
    traffic_seed=DEFAULT_TRAFFIC_SEED,
    dujo_steps=DEFAULT_DUJO_STEPS,
    dujo_traffic_mode=DEFAULT_DUJO_TRAFFIC_MODE,
    dujo_eval_mode=DEFAULT_DUJO_EVAL_MODE,
    include_simulation=False,
):
    return evaluate_method(
        method_name,
        ts,
        sat_array,
        offset_minutes=0,
        offset_seconds=offset_seconds,
        active_user_percentage=active_user_percentage,
        traffic_seed=traffic_seed,
        dujo_steps=dujo_steps,
        dujo_traffic_mode=dujo_traffic_mode,
        dujo_eval_mode=dujo_eval_mode,
        include_simulation=include_simulation,
    )
