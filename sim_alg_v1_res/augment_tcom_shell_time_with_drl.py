#!/usr/bin/env python3

import argparse
import json
import os
import time
from datetime import datetime

import pandas as pd

from tcom_revision_common import (
    HEADLINE_METHODS,
    current_paper_drl_variant,
    evaluate_method,
    load_shell_constellation,
    resolve_drl_model_path,
    snapshot_datetime,
)


def latest_dir_with_results(parent_dir, prefix=None):
    candidates = []
    for entry in os.listdir(parent_dir):
        path = os.path.join(parent_dir, entry)
        if not os.path.isdir(path):
            continue
        if prefix is not None and not entry.startswith(prefix):
            continue
        if not os.path.isfile(os.path.join(path, "results.csv")):
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No matching result directories were found under {parent_dir}")
    return max(candidates, key=os.path.getmtime)


def load_metadata(run_dir):
    metadata_path = os.path.join(run_dir, "shell_metadata.json")
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-scaling-input",
        default=None,
        help="Existing merged load-scaling results directory. Defaults to the latest merged-* directory.",
    )
    parser.add_argument(
        "--baseline-active-user-percentage",
        type=float,
        default=1e-4,
        help="Active-user percentage used for the shell-time baseline figure.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.realpath(__file__))
    load_scaling_parent = os.path.join(base_dir, "test_tcom_shell_time_load_scaling")
    baselines_parent = os.path.join(base_dir, "test_tcom_shell_time_baselines")

    load_scaling_input = args.load_scaling_input
    if not load_scaling_input:
        load_scaling_input = latest_dir_with_results(load_scaling_parent, prefix="merged-")
    load_scaling_input = os.path.abspath(load_scaling_input)

    base_df = pd.read_csv(os.path.join(load_scaling_input, "results.csv"))
    base_metadata = load_metadata(load_scaling_input)

    ts, valid_satellites, sat_array, shell_metadata = load_shell_constellation(n_sat=1000)
    drl_variant = current_paper_drl_variant()
    drl_model_path = resolve_drl_model_path(variant=drl_variant)

    unique_points = (
        base_df[["active_user_percentage", "offset_minutes"]]
        .drop_duplicates()
        .sort_values(["active_user_percentage", "offset_minutes"], kind="stable")
    )

    drl_rows = []
    for row in unique_points.itertuples(index=False):
        active_user_percentage = float(row.active_user_percentage)
        offset_minutes = int(row.offset_minutes)
        tic = time.perf_counter()
        result = evaluate_method(
            "DRL",
            ts,
            sat_array,
            offset_minutes=offset_minutes,
            active_user_percentage=active_user_percentage,
        )
        elapsed = time.perf_counter() - tic
        drl_rows.append(
            {
                "offset_minutes": offset_minutes,
                "snapshot_utc": snapshot_datetime(offset_minutes).isoformat(),
                "active_user_percentage": active_user_percentage,
                "method": "DRL",
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

    drl_df = pd.DataFrame(drl_rows)
    merged_df = pd.concat(
        [
            base_df[~base_df["method"].isin(["LD", "DeepLaDu", "DRL"])].copy(),
            drl_df,
        ],
        ignore_index=True,
    )
    merged_df = merged_df.sort_values(
        ["active_user_percentage", "offset_minutes", "method"],
        kind="stable",
    )

    timestamp = datetime.now().strftime("%Y-%B-%d-%H-%M-%S")

    load_scaling_output = os.path.join(load_scaling_parent, f"merged-{timestamp}-ail")
    os.makedirs(load_scaling_output, exist_ok=True)
    merged_df.to_csv(os.path.join(load_scaling_output, "results.csv"), index=False)

    updated_load_metadata = dict(base_metadata)
    updated_load_metadata.update(shell_metadata)
    updated_load_metadata["methods"] = HEADLINE_METHODS
    updated_load_metadata["drl_variant"] = drl_variant
    updated_load_metadata["drl_model_path"] = drl_model_path
    updated_load_metadata["augmented_from"] = load_scaling_input
    with open(os.path.join(load_scaling_output, "shell_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(updated_load_metadata, fp, indent=2)

    baseline_df = merged_df[
        merged_df["active_user_percentage"] == args.baseline_active_user_percentage
    ].copy()
    baseline_df = baseline_df[baseline_df["method"].isin(HEADLINE_METHODS)].copy()
    baseline_df = baseline_df.sort_values(["offset_minutes", "method"], kind="stable")

    baselines_output = os.path.join(
        baselines_parent,
        f"test_tcom_shell_time_baselines-{timestamp}-drl-ail",
    )
    os.makedirs(baselines_output, exist_ok=True)
    baseline_df.to_csv(os.path.join(baselines_output, "results.csv"), index=False)

    baseline_metadata = dict(shell_metadata)
    baseline_metadata["active_user_percentage"] = args.baseline_active_user_percentage
    baseline_metadata["methods"] = HEADLINE_METHODS
    baseline_metadata["time_offsets_minutes"] = sorted(
        baseline_df["offset_minutes"].drop_duplicates().astype(int).tolist()
    )
    baseline_metadata["drl_variant"] = drl_variant
    baseline_metadata["drl_model_path"] = drl_model_path
    baseline_metadata["augmented_from"] = load_scaling_input
    with open(os.path.join(baselines_output, "shell_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(baseline_metadata, fp, indent=2)

    print(load_scaling_output)
    print(baselines_output)


if __name__ == "__main__":
    main()
