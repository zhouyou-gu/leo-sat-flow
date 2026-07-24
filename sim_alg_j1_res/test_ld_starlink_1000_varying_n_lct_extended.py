#!/usr/bin/env python3
"""Evaluate DeepLaDu and existing baselines across average LCT counts."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from sim_alg_j1_res.lct_experiment_helpers import (
    DEFAULT_AVERAGE_LCTS,
    METHODS,
    build_average_lct_mask,
    parse_float_list,
    parse_int_list,
    read_csv,
    summarize_sweep_rows,
    validate_sweep_rows,
    write_csv,
)
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT
from working_dir_path import get_working_dir_path


DEFAULT_SEEDS = list(range(20))
DEFAULT_N_SAT = 1000
DEFAULT_M = 5


def heuristic_arguments(method, seed):
    """Return the established matching and routing arguments for a baseline."""
    mapping = {
        "+Grid": ("grid", "ospf", seed),
        "Rand": ("rand", "ospf", seed),
        "MRate": ("mwm", "ospf", seed),
        "SaTE": ("mwm", "spf", seed),
    }
    if method not in mapping:
        raise ValueError(f"Unknown heuristic method {method}")
    return mapping[method]


def build_result_row(
    average_lcts,
    seed,
    method,
    realized_average_lcts,
    primal_objective,
    runtime_ms,
    possible_graph_connected,
):
    """Build one case-level sweep record."""
    return {
        "average_lcts": float(average_lcts),
        "seed": int(seed),
        "method": method,
        "realized_average_lcts": float(realized_average_lcts),
        "throughput_gbps": max(0.0, float(-primal_objective)),
        "runtime_ms": float(runtime_ms),
        "possible_graph_connected": bool(possible_graph_connected),
    }


def build_case(n_sat, seed, average_lcts, tle_path, model_path):
    """Build one constellation and traffic case shared by all five methods."""
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
    simulation.lct_mask = build_average_lct_mask(
        n_sat=n_sat,
        average_lcts=average_lcts,
        seed=seed,
    )
    simulation.update_space()
    simulation.set_solver(solver)
    simulation.update_solver_traffic_info(seed=seed)
    solver.load_gnn(path=model_path)
    connected, _ = solver.check_connected()
    realized = float(simulation.lct_mask.sum()) / n_sat
    return solver, realized, connected


def evaluate_method(method, solver, seed):
    """Evaluate one method on an already constructed case."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    if method == "DeepLaDu":
        solver.infer_gnn()
        primal = solver.get_prim_objective(with_rates=False)
    else:
        matching, routing, matching_seed = heuristic_arguments(method, seed)
        primal = solver.get_prim_objective_heuristic(
            with_rates=False,
            matching_method=matching,
            routing_method=routing,
            seed=matching_seed,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_ms = 1000.0 * (time.perf_counter() - start)
    return primal, runtime_ms


def run_case(n_sat, seed, average_lcts, tle_path, model_path):
    """Evaluate every method for one average-LCT level and seed."""
    solver, realized, connected = build_case(
        n_sat=n_sat,
        seed=seed,
        average_lcts=average_lcts,
        tle_path=tle_path,
        model_path=model_path,
    )
    rows = []
    for method in METHODS:
        primal, runtime_ms = evaluate_method(method, solver, seed)
        rows.append(
            build_result_row(
                average_lcts=average_lcts,
                seed=seed,
                method=method,
                realized_average_lcts=realized,
                primal_objective=primal,
                runtime_ms=runtime_ms,
                possible_graph_connected=connected,
            )
        )
    return rows


def file_sha256(path):
    """Return the SHA-256 digest of a reproducibility-critical file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(root_dir):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--average-lcts",
        default=",".join(map(str, DEFAULT_AVERAGE_LCTS)),
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


def build_metadata(root_dir, levels, seeds, args):
    script_path = os.path.realpath(__file__)
    helper_path = os.path.join(
        root_dir,
        "sim_alg_j1_res",
        "lct_experiment_helpers.py",
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root_dir,
            text=True,
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root_dir,
                text=True,
            ).strip()
        ),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "n_sat": args.n_sat,
        "m": DEFAULT_M,
        "average_lcts": levels,
        "seeds": seeds,
        "methods": list(METHODS),
        "model": os.path.relpath(args.model, root_dir),
        "model_sha256": file_sha256(args.model),
        "script_sha256": file_sha256(script_path),
        "helper_sha256": file_sha256(helper_path),
        "mask_definition": (
            "exact total enabled slots; front/back only through 2.0, "
            "front/back fixed with additional side slots above 2.0"
        ),
    }


def main():
    root_dir = get_working_dir_path()
    args = parse_args(root_dir)
    levels = [round(value, 1) for value in parse_float_list(args.average_lcts)]
    seeds = parse_int_list(args.seeds)

    if args.validate_results:
        rows = read_csv(args.validate_results)
        validate_sweep_rows(rows, levels, seeds)
        print(f"Validated {len(rows)} sweep rows")
        return
    if not levels or min(levels) < 0.0 or max(levels) > 4.0:
        raise ValueError("Average LCT levels must lie in [0, 4]")
    if len(set(levels)) != len(levels):
        raise ValueError("Average LCT levels must be unique")
    if not seeds:
        raise ValueError("At least one seed is required")
    if args.n_sat < 1:
        raise ValueError("n_sat must be positive")

    tle_path = os.path.join(root_dir, "starlink_16_jul_2025_1600.tle")
    output_dir = Path(
        args.output_dir or GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"

    if not args.skip_warmup:
        print("Running one unrecorded five-method warm-up case")
        run_case(
            n_sat=args.n_sat,
            seed=max(seeds) + 1000,
            average_lcts=levels[0],
            tle_path=tle_path,
            model_path=args.model,
        )

    rows = []
    for average_lcts in levels:
        for seed in seeds:
            print(f"Building average_lcts={average_lcts}, seed={seed}")
            case_rows = run_case(
                n_sat=args.n_sat,
                seed=seed,
                average_lcts=average_lcts,
                tle_path=tle_path,
                model_path=args.model,
            )
            rows.extend(case_rows)
            write_csv(results_path, rows)
            for row in case_rows:
                print(json.dumps(row, sort_keys=True))

    validate_sweep_rows(rows, levels, seeds)
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "metadata.json"
    write_csv(summary_path, summarize_sweep_rows(rows))
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(
            build_metadata(root_dir, levels, seeds, args),
            fp,
            indent=2,
        )

    print(f"Validated {len(rows)} sweep rows")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
