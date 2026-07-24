#!/usr/bin/env python3
"""Measure DeepLaDu congestion prices for exact two-, three-, and four-LCT layouts."""

import argparse
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from sim_alg_j1_res.lct_experiment_helpers import (
    build_exact_lct_mask,
    parse_int_list,
    selected_price_sample_rows,
    write_csv,
)
from sim_alg_j1_res.test_ld_starlink_1000_varying_n_lct_extended import (
    DEFAULT_M,
    DEFAULT_N_SAT,
    DEFAULT_SEEDS,
    file_sha256,
)
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT
from working_dir_path import get_working_dir_path


DEFAULT_LCT_COUNTS = (2, 3, 4)
DEFAULT_FOR_THETA_HALF = 60.0


def constellation_ratio(constellation):
    """Map the supported constellation name to its regular-Walker ratio."""
    ratios = {
        "starlink": 0.0,
        "walker-delta": 1.0,
    }
    if constellation not in ratios:
        raise ValueError(f"Unsupported constellation {constellation}")
    return ratios[constellation]


def configure_field_of_regard(simulation, theta_half):
    """Configure one simulation instance with the requested FOR half-angle."""
    theta_half = float(theta_half)
    if not 0.0 < theta_half <= 90.0:
        raise ValueError("theta_half must lie in (0, 90]")
    simulation.FOR_THETA_HALF = theta_half
    simulation.cos_threshold = math.cos(math.radians(theta_half))


def build_result_row(
    lct_count,
    seed,
    possible_graph_connected,
    lengths,
    prices,
    primal_objective,
):
    """Build one realization-level exact-LCT congestion-price record."""
    lengths = np.asarray(lengths, dtype=np.int64)
    prices = np.asarray(prices, dtype=np.float64)
    reachable = lengths > 0
    if not reachable.any():
        raise ValueError("No selected routes are reachable")
    if prices.ndim != 1 or not len(prices):
        raise ValueError("Selected endpoint prices must be non-empty")
    return {
        "lct_count": int(lct_count),
        "seed": int(seed),
        "realized_average_lcts": float(lct_count),
        "possible_graph_connected": bool(possible_graph_connected),
        "selected_route_reachability": float(np.mean(reachable)),
        "selected_endpoint_count": int(len(prices)),
        "congestion_price_mean": float(np.mean(prices)),
        "throughput_gbps": max(0.0, float(-primal_objective)),
        "average_route_hops": float(np.mean(lengths[reachable] - 1)),
    }


def build_case(
    n_sat,
    seed,
    lct_count,
    tle_path,
    model_path,
    constellation="walker-delta",
    for_theta_half=DEFAULT_FOR_THETA_HALF,
):
    """Build one exact-LCT constellation and traffic case."""
    from sim_alg_j1_res.test_ld_starlink_1000_sg_compare import (
        ldl_sg_compare_solver,
    )
    from sim_alg_j1_res.train_rl_starlink_1000_ld import GNNSimulation
    from sim_mld.tle import generate_tle_partly_regular_constellation1000

    solver = ldl_sg_compare_solver()
    solver.N_NEAREST_GATEWAY_SATS = DEFAULT_M
    ts, _, sat_array = generate_tle_partly_regular_constellation1000(
        n_sat=n_sat,
        ratio=constellation_ratio(constellation),
        starlink_tle_path=tle_path,
        seed=seed,
    )
    simulation = GNNSimulation(ts, sat_array)
    configure_field_of_regard(simulation, for_theta_half)
    simulation.lct_mask = build_exact_lct_mask(
        n_sat=n_sat,
        lct_count=lct_count,
        seed=seed,
    )
    simulation.update_space()
    simulation.set_solver(solver)
    simulation.update_solver_traffic_info(seed=seed)
    solver.load_gnn(path=model_path)
    connected, _ = solver.check_connected()
    realized = float(simulation.lct_mask.sum()) / n_sat
    if not np.isclose(realized, lct_count):
        raise ValueError(
            f"Expected an exact {lct_count}-LCT mask, found {realized}"
        )
    return solver, connected


def run_case(
    n_sat,
    seed,
    lct_count,
    tle_path,
    model_path,
    constellation="walker-delta",
    for_theta_half=DEFAULT_FOR_THETA_HALF,
):
    """Infer prices and evaluate the selected topology for one realization."""
    solver, possible_graph_connected = build_case(
        n_sat=n_sat,
        seed=seed,
        lct_count=lct_count,
        tle_path=tle_path,
        model_path=model_path,
        constellation=constellation,
        for_theta_half=for_theta_half,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    solver.infer_gnn()
    (
        primal,
        _,
        _,
        (_, lengths, _),
        connected_sat,
        connected_lct,
    ) = solver.get_prim_objective(with_rates=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    forward_prices = solver.price_graph.get_prices(connected_sat)
    reverse_prices = solver.price_graph.get_prices(connected_sat[:, ::-1])
    sample_rows = selected_price_sample_rows(
        connected_lct=connected_lct,
        forward_prices=forward_prices,
        reverse_prices=reverse_prices,
    )
    prices = [row["price"] for row in sample_rows]
    result_row = build_result_row(
        lct_count=lct_count,
        seed=seed,
        possible_graph_connected=possible_graph_connected,
        lengths=lengths,
        prices=prices,
        primal_objective=primal,
    )
    price_samples = [
        {"lct_count": lct_count, "seed": seed, **sample}
        for sample in sample_rows
    ]
    return result_row, price_samples


def validate_rows(rows, price_samples, lct_counts, seeds):
    """Validate exact-LCT result and selected-endpoint sample completeness."""
    expected_keys = {
        (lct_count, seed)
        for lct_count in lct_counts
        for seed in seeds
    }
    row_keys = {(row["lct_count"], row["seed"]) for row in rows}
    if len(rows) != len(expected_keys) or row_keys != expected_keys:
        raise ValueError("Exact-LCT result matrix is incomplete or duplicated")
    if not price_samples:
        raise ValueError("The selected-endpoint price table is empty")

    samples_by_key = {}
    for sample in price_samples:
        key = (sample["lct_count"], sample["seed"])
        if key not in expected_keys:
            raise ValueError(f"Unexpected price-sample key {key}")
        samples_by_key.setdefault(key, []).append(sample)

    rows_by_key = {
        (row["lct_count"], row["seed"]): row
        for row in rows
    }
    for key in expected_keys:
        row = rows_by_key[key]
        samples = samples_by_key.get(key, [])
        if not row["possible_graph_connected"]:
            raise ValueError(f"Disconnected possible graph for {key}")
        if not np.isclose(row["realized_average_lcts"], key[0]):
            raise ValueError(f"Incorrect realized LCT count for {key}")
        if len(samples) != row["selected_endpoint_count"]:
            raise ValueError(f"Selected-endpoint count mismatch for {key}")
        if not np.isclose(
            np.mean([sample["price"] for sample in samples]),
            row["congestion_price_mean"],
        ):
            raise ValueError(f"Congestion-price mean mismatch for {key}")


def summarize_rows(rows):
    """Aggregate result metrics by exact LCT count."""
    summary = []
    for lct_count in sorted({row["lct_count"] for row in rows}):
        selected = [row for row in rows if row["lct_count"] == lct_count]
        summary.append(
            {
                "lct_count": lct_count,
                "n_cases": len(selected),
                "throughput_gbps_mean": float(
                    np.mean([row["throughput_gbps"] for row in selected])
                ),
                "selected_route_reachability_mean": float(
                    np.mean(
                        [
                            row["selected_route_reachability"]
                            for row in selected
                        ]
                    )
                ),
                "congestion_price_mean": float(
                    np.mean(
                        [row["congestion_price_mean"] for row in selected]
                    )
                ),
                "average_route_hops_mean": float(
                    np.mean([row["average_route_hops"] for row in selected])
                ),
            }
        )
    return summary


def parse_args(root_dir):
    """Parse the exact-LCT experiment configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lct-counts",
        default=",".join(map(str, DEFAULT_LCT_COUNTS)),
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--n-sat", type=int, default=DEFAULT_N_SAT)
    parser.add_argument(
        "--constellation",
        choices=("starlink", "walker-delta"),
        default="walker-delta",
    )
    parser.add_argument(
        "--for-theta-half",
        type=float,
        default=DEFAULT_FOR_THETA_HALF,
        help="LCT field-of-regard half-angle in degrees",
    )
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
    return parser.parse_args()


def build_metadata(root_dir, lct_counts, seeds, args):
    """Build a reproducibility record for the exact-LCT experiment."""
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
        "constellation": args.constellation,
        "regular_walker_ratio": constellation_ratio(args.constellation),
        "for_theta_half_deg": args.for_theta_half,
        "for_total_span_deg": 2.0 * args.for_theta_half,
        "m": DEFAULT_M,
        "lct_counts": lct_counts,
        "seeds": seeds,
        "model": os.path.relpath(args.model, root_dir),
        "model_sha256": file_sha256(args.model),
        "script_sha256": file_sha256(script_path),
        "helper_sha256": file_sha256(helper_path),
        "terminal_mapping": {
            "0": "front",
            "1": "back",
            "2": "right",
            "3": "left",
        },
        "lct_layouts": {
            "2": "front and back",
            "3": (
                "front and back with one balanced right- or left-facing "
                "side terminal"
            ),
            "4": "front, back, right, and left",
        },
        "price_definition": (
            "arithmetic mean of the two directed DeepLaDu congestion "
            "prices on each selected LISL, assigned to both endpoint LCTs"
        ),
    }


def main():
    """Run the exact-LCT congestion-price experiment."""
    root_dir = get_working_dir_path()
    args = parse_args(root_dir)
    lct_counts = parse_int_list(args.lct_counts)
    seeds = parse_int_list(args.seeds)
    if sorted(set(lct_counts)) != list(DEFAULT_LCT_COUNTS):
        raise ValueError("lct_counts must contain exactly 2, 3, and 4")
    if not seeds:
        raise ValueError("At least one seed is required")
    if args.n_sat < 2 or args.n_sat % 2:
        raise ValueError("n_sat must be an even integer of at least two")

    tle_path = os.path.join(root_dir, "starlink_16_jul_2025_1600.tle")
    output_dir = Path(
        args.output_dir or GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    price_samples_path = output_dir / "price_samples.csv"

    if not args.skip_warmup:
        print("Running one unrecorded exact-LCT warm-up case")
        run_case(
            n_sat=args.n_sat,
            seed=max(seeds) + 1000,
            lct_count=lct_counts[0],
            tle_path=tle_path,
            model_path=args.model,
            constellation=args.constellation,
            for_theta_half=args.for_theta_half,
        )

    rows = []
    price_samples = []
    for lct_count in lct_counts:
        for seed in seeds:
            print(f"Building exact {lct_count}-LCT seed={seed}")
            row, case_price_samples = run_case(
                n_sat=args.n_sat,
                seed=seed,
                lct_count=lct_count,
                tle_path=tle_path,
                model_path=args.model,
                constellation=args.constellation,
                for_theta_half=args.for_theta_half,
            )
            rows.append(row)
            price_samples.extend(case_price_samples)
            write_csv(results_path, rows)
            write_csv(price_samples_path, price_samples)
            print(json.dumps(row, sort_keys=True))

    validate_rows(rows, price_samples, lct_counts, seeds)
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "metadata.json"
    write_csv(summary_path, summarize_rows(rows))
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(
            build_metadata(root_dir, lct_counts, seeds, args),
            fp,
            indent=2,
        )

    print(f"Validated {len(rows)} exact-LCT result rows")
    print(f"Validated {len(price_samples)} selected-endpoint price samples")
    print(f"Results: {results_path}")
    print(f"Price samples: {price_samples_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
