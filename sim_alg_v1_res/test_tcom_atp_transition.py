#!/usr/bin/env python3

import csv
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sim_mld.solver import build_csr, construct_edges_matrix_all_in_one, multi_dijkstra_with_paths
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

from tcom_revision_common import (
    DEFAULT_ACTIVE_USER_PERCENTAGES,
    DEFAULT_ATP_EVALUATION_SECONDS,
    DEFAULT_ATP_SNAPSHOT_INTERVAL_SEC,
    DEFAULT_ATP_TIME_SEC,
    DEFAULT_ATP_TIME_SWEEP_SEC,
    DEFAULT_DUJO_EVAL_MODE,
    DEFAULT_DUJO_STEPS,
    DEFAULT_DUJO_TRAFFIC_MODE,
    DEFAULT_TRAFFIC_SEED,
    HEADLINE_METHODS,
    HEURISTIC_METHOD_SPECS,
    SUPPORTING_METHODS,
    current_paper_drl_variant,
    evaluate_method_seconds,
    load_shell_constellation,
    parse_env_choice,
    parse_env_float_list,
    parse_env_int,
    parse_env_int_list,
    save_shell_metadata,
    snapshot_datetime_seconds,
)


SNAPSHOT_INTERVAL_SEC = DEFAULT_ATP_SNAPSHOT_INTERVAL_SEC
ATP_TIME_SEC = DEFAULT_ATP_TIME_SEC
ATP_TIME_SWEEP_SEC = DEFAULT_ATP_TIME_SWEEP_SEC
EVALUATION_SECONDS = DEFAULT_ATP_EVALUATION_SECONDS


def default_offsets():
    return list(range(0, EVALUATION_SECONDS + SNAPSHOT_INTERVAL_SEC, SNAPSHOT_INTERVAL_SEC))


def default_atp_times():
    return list(ATP_TIME_SWEEP_SEC)


def parse_methods(default):
    value = os.getenv("TCOM_REVISION_METHODS", "").strip()
    valid_methods = set(HEADLINE_METHODS) | set(SUPPORTING_METHODS)
    if not value:
        return list(default)
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(methods) - valid_methods)
    if unknown:
        raise ValueError(f"Unknown ATP benchmark methods: {unknown}")
    return methods


def get_output_dir():
    configured_dir = os.getenv("TCOM_REVISION_OUTPUT_DIR", "").strip()
    if configured_dir:
        return configured_dir
    return GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)


def prepare_resume_results(results_path, offsets, atp_times, fieldnames):
    if os.getenv("TCOM_REVISION_RESUME", "1") == "0" or not os.path.exists(results_path):
        return set()

    rows = []
    method_cases = {}
    with open(results_path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None or "atp_time_seconds" not in reader.fieldnames:
            with open(results_path, "w", newline="", encoding="utf-8") as out_fp:
                writer = csv.DictWriter(out_fp, fieldnames=fieldnames)
                writer.writeheader()
            return set()
        for row in reader:
            rows.append(row)
            method_cases.setdefault(row["method"], set()).add(
                (int(row["offset_seconds"]), int(row["atp_time_seconds"]))
            )

    required_cases = {
        (int(offset_seconds), int(atp_time_seconds))
        for offset_seconds in offsets
        for atp_time_seconds in atp_times
    }
    completed_methods = {
        method
        for method, seen_cases in method_cases.items()
        if required_cases.issubset(seen_cases)
    }
    kept_rows = [row for row in rows if row["method"] in completed_methods]
    if len(kept_rows) != len(rows):
        with open(results_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

    return completed_methods


def canonical_link_rows(connected_lct):
    if np.asarray(connected_lct).size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    links = np.asarray(connected_lct, dtype=np.int64)
    links = np.sort(links, axis=1)
    return links


def rows_to_link_set(link_rows):
    return {tuple(row) for row in np.asarray(link_rows, dtype=np.int64)}


def link_set_to_rows(link_set):
    if not link_set:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(link_set), dtype=np.int64).reshape((-1, 2))


def update_atp_state(offset_seconds, current_links, previous_links, first_seen, initialization, atp_time_seconds):
    if initialization:
        for link in current_links:
            first_seen[link] = offset_seconds - atp_time_seconds
        usable_links = set(current_links)
        return usable_links, set(), first_seen

    dropped_links = set(first_seen) - current_links
    for link in dropped_links:
        del first_seen[link]

    new_links = current_links - previous_links
    for link in new_links:
        first_seen[link] = offset_seconds

    for link in current_links:
        if link not in first_seen:
            first_seen[link] = offset_seconds - atp_time_seconds

    usable_links = {
        link
        for link in current_links
        if offset_seconds - first_seen[link] >= atp_time_seconds
    }
    return usable_links, new_links, first_seen


def reroute_and_allocate_with_usable_links(simulation, method, usable_lct):
    solver = simulation.solver
    if np.asarray(usable_lct).size == 0:
        rates = np.zeros(solver.data_source.shape[0], dtype=np.float64)
        lengths = np.zeros(solver.data_source.shape[0], dtype=np.int64)
        return rates, lengths

    connected_sat = usable_lct // solver.N_LCT_PER_SAT
    distance = np.linalg.norm(
        solver.positions[connected_sat[:, 0]] - solver.positions[connected_sat[:, 1]],
        axis=1,
    )
    capacity = solver.compute_capacity(distance)

    if method in {"DuJo", "DRL"}:
        edge_in_both_direction = np.concatenate((connected_sat, connected_sat[:, ::-1]), axis=0)
        prices = solver.price_graph.get_prices(edge_in_both_direction)
        indptr, indices, data = build_csr(
            solver.n_sat,
            edge_in_both_direction,
            prices,
            sym_half=False,
        )
    else:
        spec = HEURISTIC_METHOD_SPECS[method]
        if spec["routing_method"] == "ospf":
            edge_weights = 1 / capacity
        else:
            edge_weights = np.ones_like(capacity)
        indptr, indices, data = build_csr(
            solver.n_sat,
            connected_sat,
            edge_weights,
            sym_half=True,
        )

    costs, lengths, paths_all = multi_dijkstra_with_paths(
        solver.n_sat,
        indptr,
        indices,
        solver.data_source,
        solver.data_target,
        data,
    )
    srouting = construct_edges_matrix_all_in_one(
        solver.data_source,
        solver.data_target,
        costs,
        lengths,
        paths_all,
    )
    rates = solver.get_rates_prim(srouting, usable_lct, mode=solver.objective_mode)
    return rates, lengths


def atp_metrics(simulation, rates, lengths):
    offered = float(np.sum(simulation.solver.forward_traffic_demand))
    served = float(np.sum(rates))
    unmet = max(offered - served, 0.0)
    positive_lengths = lengths[lengths > 0]
    return {
        "atp_served_throughput_gbps": served,
        "atp_served_ratio": 1.0 if offered <= 0 else served / offered,
        "atp_unmet_demand_gbps": unmet,
        "atp_unmet_fraction": 0.0 if offered <= 0 else unmet / offered,
        "atp_connected_flows": int(np.count_nonzero(rates > 1e-9)),
        "atp_avg_hops": float(np.mean(positive_lengths)) if positive_lengths.size else 0.0,
    }


def safe_mean(values):
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def write_summary(results_path, summary_path):
    if not os.path.exists(results_path):
        return

    rows = []
    with open(results_path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if int(row["initialization_snapshot"]) == 0:
                rows.append(row)

    groups = {}
    for row in rows:
        key = (row["method"], int(row["atp_time_seconds"]))
        groups.setdefault(key, []).append(row)

    fieldnames = [
        "method",
        "atp_time_seconds",
        "snapshot_count",
        "avg_original_served_throughput_gbps",
        "avg_atp_served_throughput_gbps",
        "throughput_retention_ratio",
        "avg_original_served_ratio",
        "avg_atp_served_ratio",
        "avg_usable_link_fraction",
        "avg_new_link_fraction",
        "avg_acquiring_links",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for method, atp_time_seconds in sorted(groups, key=lambda item: (item[0][0], item[0][1])):
            group_rows = groups[(method, atp_time_seconds)]
            avg_original_throughput = safe_mean(
                float(row["original_served_throughput_gbps"]) for row in group_rows
            )
            avg_atp_throughput = safe_mean(
                float(row["atp_served_throughput_gbps"]) for row in group_rows
            )
            usable_link_fractions = [
                0.0
                if int(row["matched_links"]) == 0
                else int(row["usable_links"]) / int(row["matched_links"])
                for row in group_rows
            ]
            writer.writerow(
                {
                    "method": method,
                    "atp_time_seconds": atp_time_seconds,
                    "snapshot_count": len(group_rows),
                    "avg_original_served_throughput_gbps": avg_original_throughput,
                    "avg_atp_served_throughput_gbps": avg_atp_throughput,
                    "throughput_retention_ratio": (
                        0.0
                        if avg_original_throughput <= 0
                        else avg_atp_throughput / avg_original_throughput
                    ),
                    "avg_original_served_ratio": safe_mean(
                        float(row["original_served_ratio"]) for row in group_rows
                    ),
                    "avg_atp_served_ratio": safe_mean(
                        float(row["atp_served_ratio"]) for row in group_rows
                    ),
                    "avg_usable_link_fraction": safe_mean(usable_link_fractions),
                    "avg_new_link_fraction": safe_mean(
                        float(row["new_link_fraction"]) for row in group_rows
                    ),
                    "avg_acquiring_links": safe_mean(
                        int(row["acquiring_links"]) for row in group_rows
                    ),
                }
            )


def main():
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    ts, valid_satellites, sat_array, metadata = load_shell_constellation(n_sat=1000)
    offsets = parse_env_int_list("TCOM_REVISION_ATP_OFFSETS_SEC", default_offsets())
    atp_times = parse_env_int_list("TCOM_REVISION_ATP_TIMES_SEC", default_atp_times())
    active_user_percentage = parse_env_float_list(
        "TCOM_REVISION_BASE_ACTIVE_USER_PERCENTAGES",
        [DEFAULT_ACTIVE_USER_PERCENTAGES[4]],
    )[0]
    methods = parse_methods(HEADLINE_METHODS)
    dujo_steps = parse_env_int("TCOM_REVISION_DUJO_STEPS", DEFAULT_DUJO_STEPS)
    dujo_traffic_mode = parse_env_choice(
        "TCOM_REVISION_DUJO_TRAFFIC_MODE",
        DEFAULT_DUJO_TRAFFIC_MODE,
        {"legacy", "fixed"},
    )
    dujo_eval_mode = parse_env_choice(
        "TCOM_REVISION_DUJO_EVAL_MODE",
        DEFAULT_DUJO_EVAL_MODE,
        {"last", "best"},
    )

    metadata["atp_snapshot_interval_seconds"] = SNAPSHOT_INTERVAL_SEC
    metadata["atp_time_seconds_reference"] = ATP_TIME_SEC
    metadata["atp_time_sweep_seconds"] = atp_times
    metadata["time_offsets_seconds"] = offsets
    metadata["active_user_percentage"] = active_user_percentage
    metadata["dujo_steps"] = dujo_steps
    metadata["dujo_traffic_mode"] = dujo_traffic_mode
    metadata["dujo_eval_mode"] = dujo_eval_mode
    metadata["drl_variant"] = current_paper_drl_variant()
    metadata["drl_model_path"] = os.getenv("TCOM_REVISION_DRL_MODEL_PATH", "").strip() or None
    metadata["methods"] = methods
    save_shell_metadata(output_dir, metadata)

    results_path = os.path.join(output_dir, "results.csv")
    summary_path = os.path.join(output_dir, "summary.csv")
    fieldnames = [
        "offset_seconds",
        "snapshot_utc",
        "initialization_snapshot",
        "atp_time_seconds",
        "active_user_percentage",
        "method",
        "primal_objective",
        "dual_objective",
        "offered_demand_gbps",
        "original_served_throughput_gbps",
        "original_served_ratio",
        "atp_served_throughput_gbps",
        "atp_served_ratio",
        "atp_unmet_demand_gbps",
        "atp_unmet_fraction",
        "atp_connected_flows",
        "atp_avg_hops",
        "matched_links",
        "usable_links",
        "acquiring_links",
        "new_links",
        "new_link_fraction",
        "original_evaluation_seconds",
        "atp_reallocation_seconds",
        "evaluation_seconds",
    ]

    completed_methods = prepare_resume_results(results_path, offsets, atp_times, fieldnames)
    open_mode = "a" if completed_methods else "w"
    write_header = open_mode == "w" or not os.path.exists(results_path) or os.path.getsize(results_path) == 0

    with open(results_path, open_mode, newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for method in methods:
            if method in completed_methods:
                continue
            atp_states = {
                int(atp_time_seconds): {"first_seen": {}, "previous_links": set()}
                for atp_time_seconds in atp_times
            }
            for offset_index, offset_seconds in enumerate(offsets):
                original_tic = time.perf_counter()
                result = evaluate_method_seconds(
                    method,
                    ts,
                    sat_array,
                    offset_seconds=offset_seconds,
                    active_user_percentage=active_user_percentage,
                    traffic_seed=DEFAULT_TRAFFIC_SEED,
                    dujo_steps=dujo_steps,
                    dujo_traffic_mode=dujo_traffic_mode,
                    dujo_eval_mode=dujo_eval_mode,
                    include_simulation=True,
                )
                original_elapsed = time.perf_counter() - original_tic
                simulation = result.pop("_simulation")
                current_lct = canonical_link_rows(result["connected_lct"])
                current_links = rows_to_link_set(current_lct)
                initialization_snapshot = offset_index == 0
                matched_links = len(current_links)
                for atp_time_seconds in atp_times:
                    atp_time_seconds = int(atp_time_seconds)
                    state = atp_states[atp_time_seconds]
                    usable_links, new_links, first_seen = update_atp_state(
                        offset_seconds,
                        current_links,
                        state["previous_links"],
                        state["first_seen"],
                        initialization_snapshot,
                        atp_time_seconds,
                    )
                    state["first_seen"] = first_seen
                    state["previous_links"] = current_links
                    usable_lct = link_set_to_rows(usable_links)
                    atp_tic = time.perf_counter()
                    atp_rates, atp_lengths = reroute_and_allocate_with_usable_links(
                        simulation,
                        method,
                        usable_lct,
                    )
                    atp_elapsed = time.perf_counter() - atp_tic
                    metrics = atp_metrics(simulation, atp_rates, atp_lengths)
                    usable_link_count = len(usable_links)
                    writer.writerow(
                        {
                            "offset_seconds": offset_seconds,
                            "snapshot_utc": snapshot_datetime_seconds(offset_seconds).isoformat(),
                            "initialization_snapshot": int(initialization_snapshot),
                            "atp_time_seconds": atp_time_seconds,
                            "active_user_percentage": active_user_percentage,
                            "method": method,
                            "primal_objective": result["primal_objective"],
                            "dual_objective": result["dual_objective"],
                            "offered_demand_gbps": result["offered_demand_gbps"],
                            "original_served_throughput_gbps": result["served_throughput_gbps"],
                            "original_served_ratio": result["served_ratio"],
                            "matched_links": matched_links,
                            "usable_links": usable_link_count,
                            "acquiring_links": matched_links - usable_link_count,
                            "new_links": len(new_links),
                            "new_link_fraction": (
                                0.0 if matched_links == 0 else len(new_links) / matched_links
                            ),
                            "original_evaluation_seconds": original_elapsed,
                            "atp_reallocation_seconds": atp_elapsed,
                            "evaluation_seconds": original_elapsed + atp_elapsed,
                            **metrics,
                        }
                    )
                fp.flush()
    write_summary(results_path, summary_path)


if __name__ == "__main__":
    main()
