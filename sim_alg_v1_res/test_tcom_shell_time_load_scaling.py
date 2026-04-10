#!/usr/bin/env python3

import csv
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

from tcom_revision_common import (
    DEFAULT_ACTIVE_USER_PERCENTAGES,
    DEFAULT_DUJO_STEPS,
    DEFAULT_DUJO_EVAL_MODE,
    DEFAULT_DUJO_TRAFFIC_MODE,
    DEFAULT_TIME_OFFSETS_MINUTES,
    HEADLINE_METHODS,
    current_paper_drl_variant,
    evaluate_method,
    load_shell_constellation,
    parse_env_choice,
    parse_env_float_list,
    parse_env_int,
    parse_env_int_list,
    save_shell_metadata,
    snapshot_datetime,
)


def main():
    output_dir = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
    os.makedirs(output_dir, exist_ok=True)

    ts, valid_satellites, sat_array, metadata = load_shell_constellation(n_sat=1000)
    time_offsets = parse_env_int_list("TCOM_REVISION_TIME_OFFSETS_MIN", DEFAULT_TIME_OFFSETS_MINUTES)
    active_user_percentages = parse_env_float_list(
        "TCOM_REVISION_ACTIVE_USER_PERCENTAGES",
        DEFAULT_ACTIVE_USER_PERCENTAGES,
    )
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

    metadata["time_offsets_minutes"] = time_offsets
    metadata["active_user_percentages"] = active_user_percentages
    metadata["dujo_steps"] = dujo_steps
    metadata["dujo_traffic_mode"] = dujo_traffic_mode
    metadata["dujo_eval_mode"] = dujo_eval_mode
    metadata["drl_variant"] = current_paper_drl_variant()
    metadata["drl_model_path"] = os.getenv("TCOM_REVISION_DRL_MODEL_PATH", "").strip() or None
    metadata["methods"] = HEADLINE_METHODS
    save_shell_metadata(output_dir, metadata)

    results_path = os.path.join(output_dir, "results.csv")
    fieldnames = [
        "offset_minutes",
        "snapshot_utc",
        "active_user_percentage",
        "method",
        "primal_objective",
        "dual_objective",
        "offered_demand_gbps",
        "served_throughput_gbps",
        "served_ratio",
        "unmet_demand_gbps",
        "unmet_fraction",
        "connected_flows",
        "avg_hops",
        "matched_satellite_pairs",
        "evaluation_seconds",
    ]

    with open(results_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for active_user_percentage in active_user_percentages:
            for offset_minutes in time_offsets:
                for method in HEADLINE_METHODS:
                    tic = time.perf_counter()
                    result = evaluate_method(
                        method,
                        ts,
                        sat_array,
                        offset_minutes=offset_minutes,
                        active_user_percentage=active_user_percentage,
                        dujo_steps=dujo_steps,
                        dujo_traffic_mode=dujo_traffic_mode,
                        dujo_eval_mode=dujo_eval_mode,
                    )
                    elapsed = time.perf_counter() - tic
                    writer.writerow(
                        {
                            "offset_minutes": offset_minutes,
                            "snapshot_utc": snapshot_datetime(offset_minutes).isoformat(),
                            "active_user_percentage": active_user_percentage,
                            "method": method,
                            "primal_objective": result["primal_objective"],
                            "dual_objective": result["dual_objective"],
                            "offered_demand_gbps": result["offered_demand_gbps"],
                            "served_throughput_gbps": result["served_throughput_gbps"],
                            "served_ratio": result["served_ratio"],
                            "unmet_demand_gbps": result["unmet_demand_gbps"],
                            "unmet_fraction": result["unmet_fraction"],
                            "connected_flows": result["connected_flows"],
                            "avg_hops": result["avg_hops"],
                            "matched_satellite_pairs": result["matched_satellite_pairs"],
                            "evaluation_seconds": elapsed,
                        }
                    )
                    fp.flush()


if __name__ == "__main__":
    main()
