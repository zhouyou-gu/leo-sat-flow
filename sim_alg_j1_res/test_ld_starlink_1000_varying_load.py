#!/usr/bin/env python3
"""Evaluate DeepLaDu and existing baselines under steady traffic-load scaling."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch

from sim_alg_j1_res.varying_load_helpers import (
    METHODS,
    parse_float_list,
    parse_int_list,
    read_csv,
    summarize_rows,
    validate_rows,
    write_csv,
)
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT
from working_dir_path import get_working_dir_path


DEFAULT_ACTIVE_USER_PERCENTAGES = [
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
]
DEFAULT_SEEDS = list(range(20))
DEFAULT_N_SAT = 1000
DEFAULT_M = 5


def heuristic_arguments(method, seed):
    """Return the established matching/routing arguments for a baseline."""
    mapping = {
        "MRate": ("mwm", "ospf", seed),
        "+Grid": ("grid", "ospf", seed),
        "Rand": ("rand", "ospf", seed),
        "SaTE": ("mwm", "spf", seed),
    }
    if method not in mapping:
        raise ValueError(f"Unknown heuristic method: {method}")
    return mapping[method]


def build_result_row(
    active_user_percentage,
    seed,
    method,
    offered_demand_gbps,
    primal_objective,
    runtime_ms,
    flow_pair_count,
    demand_satellite_count,
    matched_satellite_pair_count,
    rates,
):
    """Build one validated case record from solver outputs."""
    served_throughput_gbps = max(0.0, float(-primal_objective))
    served_ratio = (
        served_throughput_gbps / offered_demand_gbps
        if offered_demand_gbps > 0
        else 0.0
    )
    return {
        "active_user_percentage": active_user_percentage,
        "seed": seed,
        "method": method,
        "offered_demand_gbps": offered_demand_gbps,
        "served_throughput_gbps": served_throughput_gbps,
        "served_ratio": served_ratio,
        "runtime_ms": runtime_ms,
        "flow_pair_count": flow_pair_count,
        "demand_satellite_count": demand_satellite_count,
        "matched_satellite_pair_count": matched_satellite_pair_count,
        "allocated_flow_count": int(np.count_nonzero(rates > 0)),
    }


def build_case(n_sat, seed, active_user_percentage, tle_path):
    """Build one deterministic constellation/traffic case shared by all methods."""
    from sim_alg_j1_res.test_ld_starlink_1000_sg_compare import (
        ldl_sg_compare_solver,
    )
    from sim_alg_j1_res.train_rl_starlink_1000_ld import GNNSimulation
    from sim_mld.tle import generate_tle_partly_regular_constellation1000

    solver = ldl_sg_compare_solver()
    solver.N_NEAREST_GATEWAY_SATS = DEFAULT_M
    ts, _, sat_array = generate_tle_partly_regular_constellation1000(
        n_sat=n_sat,
        ratio=0.0,
        starlink_tle_path=tle_path,
        seed=seed,
    )
    simulation = GNNSimulation(ts, sat_array)
    simulation.terrain.ACTIVE_USER_PERCENTAGE = active_user_percentage
    simulation.config_l_mask(seed=seed)
    simulation.update_space()
    simulation.set_solver(solver)
    simulation.update_solver_traffic_info(seed=seed)
    return solver


def evaluate_method(method, solver, seed):
    """Evaluate one method on the already constructed matched case."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    if method == "DeepLaDu":
        solver.infer_gnn()
        p_o, rates, _, _, connected_sat, _ = solver.get_prim_objective(
            with_rates=True
        )
    else:
        matching_method, routing_method, matching_seed = heuristic_arguments(
            method, seed
        )
        p_o, rates, _, _, connected_sat, _ = solver.get_prim_objective_heuristic(
            with_rates=True,
            matching_method=matching_method,
            routing_method=routing_method,
            seed=matching_seed,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_ms = 1000.0 * (time.perf_counter() - start)
    return p_o, rates, connected_sat, runtime_ms


def file_sha256(path):
    """Return the SHA-256 digest for a reproducibility-critical file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(root_dir):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--active-user-percentages",
        default=",".join(map(str, DEFAULT_ACTIVE_USER_PERCENTAGES)),
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--n-sat", type=int, default=DEFAULT_N_SAT)
    parser.add_argument(
        "--model",
        default=os.path.join(
            root_dir,
            "sim_alg_j1_res",
            "selected_nn",
            "ld_model.model_final_beta_0_7000_pt.pt",
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--validate-results", default=None)
    return parser.parse_args()


def run_case(n_sat, seed, active_user_percentage, tle_path, model_path):
    """Evaluate all five methods on one matched constellation/traffic case."""
    solver = build_case(n_sat, seed, active_user_percentage, tle_path)
    solver.load_gnn(path=model_path)
    offered_demand_gbps = float(np.sum(solver.forward_traffic_demand))
    flow_pair_count = int(solver.data_source.shape[0])
    demand_satellite_count = int(
        np.count_nonzero(solver.forward_traffic_demand)
    )

    rows = []
    for method in METHODS:
        p_o, rates, connected_sat, runtime_ms = evaluate_method(method, solver, seed)
        rows.append(
            build_result_row(
                active_user_percentage=active_user_percentage,
                seed=seed,
                method=method,
                offered_demand_gbps=offered_demand_gbps,
                primal_objective=p_o,
                runtime_ms=runtime_ms,
                flow_pair_count=flow_pair_count,
                demand_satellite_count=demand_satellite_count,
                matched_satellite_pair_count=int(connected_sat.shape[0]),
                rates=rates,
            )
        )
    return rows


def main():
    root_dir = get_working_dir_path()
    args = parse_args(root_dir)
    loads = parse_float_list(args.active_user_percentages)
    seeds = parse_int_list(args.seeds)

    if args.validate_results:
        rows = read_csv(args.validate_results)
        validate_rows(rows, loads, seeds)
        print(f"Validated {len(rows)} rows")
        return
    if not loads or min(loads) <= 0:
        raise ValueError("Active-user percentages must be positive")
    if not seeds:
        raise ValueError("At least one seed is required")

    tle_path = os.path.join(root_dir, "starlink_16_jul_2025_1600.tle")
    output_dir = args.output_dir or GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_warmup:
        print("Running one unrecorded five-method warm-up case")
        run_case(
            args.n_sat,
            max(seeds) + 1000,
            loads[0],
            tle_path,
            args.model,
        )

    rows = []
    for active_user_percentage in loads:
        for seed in seeds:
            print(f"Building load={active_user_percentage}, seed={seed}")
            case_rows = run_case(
                args.n_sat,
                seed,
                active_user_percentage,
                tle_path,
                args.model,
            )
            rows.extend(case_rows)
            for row in case_rows:
                print(json.dumps(row, sort_keys=True))

    validate_rows(rows, loads, seeds)
    results_path = os.path.join(output_dir, "results.csv")
    summary_path = os.path.join(output_dir, "summary.csv")
    write_csv(results_path, rows)
    write_csv(summary_path, summarize_rows(rows, loads))

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root_dir, text=True
    ).strip()
    script_path = os.path.realpath(__file__)
    helper_path = os.path.join(
        root_dir, "sim_alg_j1_res", "varying_load_helpers.py"
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root_dir, text=True
            ).strip()
        ),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_sat": args.n_sat,
        "m": DEFAULT_M,
        "per_user_demand_gbps": 0.1,
        "active_user_percentages": loads,
        "seeds": seeds,
        "methods": list(METHODS),
        "model": os.path.relpath(args.model, root_dir),
        "model_sha256": file_sha256(args.model),
        "script_sha256": file_sha256(script_path),
        "helper_sha256": file_sha256(helper_path),
        "served_ratio_definition": (
            "served throughput divided by total residual forward demand"
        ),
        "runtime_definition": (
            "method-specific online matching, routing, and rate allocation; "
            "DeepLaDu includes GNN inference but excludes model loading"
        ),
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"Validated {len(rows)} rows")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
