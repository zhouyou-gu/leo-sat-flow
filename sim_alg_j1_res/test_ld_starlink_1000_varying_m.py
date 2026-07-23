#!/usr/bin/env python3
"""Evaluate DeepLaDu sensitivity to the number of serving candidates M."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch

from sim_alg_j1_res.test_ld_starlink_1000_sg_compare import ldl_sg_compare_solver
from sim_alg_j1_res.train_rl_starlink_1000_ld import GNNSimulation
from sim_mld.tle import generate_tle_partly_regular_constellation1000
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT
from working_dir_path import get_working_dir_path


DEFAULT_M_VALUES = [1, 3, 5, 7, 10]
DEFAULT_SEEDS = list(range(20))
DEFAULT_N_SAT = 1000


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root_dir):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root_dir, text=True
    ).strip()


def build_case(n_sat, seed, m_value, tle_path, checkpoint_path):
    solver = ldl_sg_compare_solver()
    solver.N_NEAREST_GATEWAY_SATS = m_value
    ts, _, sat_array = generate_tle_partly_regular_constellation1000(
        n_sat=n_sat,
        ratio=0.0,
        starlink_tle_path=tle_path,
        seed=seed,
    )
    simulation = GNNSimulation(ts, sat_array)
    simulation.config_l_mask(seed=seed)
    simulation.update_space()
    simulation.set_solver(solver)
    simulation.update_solver_traffic_info(seed=seed)
    solver.load_gnn(path=checkpoint_path)
    return simulation, solver


def evaluate_case(n_sat, seed, m_value, tle_path, checkpoint_path):
    simulation, solver = build_case(
        n_sat=n_sat,
        seed=seed,
        m_value=m_value,
        tle_path=tle_path,
        checkpoint_path=checkpoint_path,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    solver.infer_gnn()
    p_o, rates, _, _, _, _ = solver.get_prim_objective(with_rates=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    online_time_ms = 1000.0 * (time.perf_counter() - start)

    pair_count = int(solver.data_source.shape[0])
    demand_count = int(np.count_nonzero(solver.forward_traffic_demand))
    return {
        "seed": seed,
        "m": m_value,
        "pair_count": pair_count,
        "demand_satellite_count": demand_count,
        "pairs_per_demand_satellite": pair_count / demand_count,
        "throughput_gbps": float(-p_o),
        "online_time_ms": online_time_ms,
        "allocated_flow_count": int(np.count_nonzero(rates > 0)),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, m_values):
    summary_rows = []
    for m_value in m_values:
        selected = [row for row in rows if row["m"] == m_value]
        summary = {"m": m_value, "n_cases": len(selected)}
        for metric in (
            "pair_count",
            "pairs_per_demand_satellite",
            "throughput_gbps",
            "online_time_ms",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))
        summary_rows.append(summary)
    return summary_rows


def main():
    root_dir = get_working_dir_path()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-values", default=",".join(map(str, DEFAULT_M_VALUES)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--n-sat", type=int, default=DEFAULT_N_SAT)
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            root_dir,
            "sim_alg_j1_res",
            "selected_nn",
            "ld_model.model_final_beta_0_7000_pt.pt",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; defaults to the repository timestamped log path.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the unrecorded pipeline warm-up case.",
    )
    args = parser.parse_args()

    m_values = parse_int_list(args.m_values)
    seeds = parse_int_list(args.seeds)
    if not m_values or min(m_values) < 1:
        raise ValueError("All M values must be positive integers")
    if not seeds:
        raise ValueError("At least one seed is required")

    tle_path = os.path.join(root_dir, "starlink_16_jul_2025_1600.tle")
    output_dir = args.output_dir or GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_warmup:
        print("Running one unrecorded M=5 warm-up case")
        warmup_seed = max(seeds) + 1000
        evaluate_case(
            n_sat=args.n_sat,
            seed=warmup_seed,
            m_value=5,
            tle_path=tle_path,
            checkpoint_path=args.checkpoint,
        )

    rows = []
    for seed in seeds:
        for m_value in m_values:
            print(f"Evaluating seed={seed}, M={m_value}")
            row = evaluate_case(
                n_sat=args.n_sat,
                seed=seed,
                m_value=m_value,
                tle_path=tle_path,
                checkpoint_path=args.checkpoint,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    results_fields = list(rows[0].keys())
    results_path = os.path.join(output_dir, "results.csv")
    write_csv(results_path, rows, results_fields)

    summary_rows = summarize(rows, m_values)
    summary_fields = list(summary_rows[0].keys())
    summary_path = os.path.join(output_dir, "summary.csv")
    write_csv(summary_path, summary_rows, summary_fields)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root_dir),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_sat": args.n_sat,
        "m_values": m_values,
        "seeds": seeds,
        "checkpoint": os.path.relpath(args.checkpoint, root_dir),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "candidate_metric": "shortest-path physical distance on the current satellite graph",
        "online_time_definition": "one DeepLaDu inference plus matching, routing, and flow-rate allocation",
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")
    for row in summary_rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
