"""Pure helpers for the extended LCT-count experiments."""

import csv
from pathlib import Path

import numpy as np


METHODS = ("DeepLaDu", "+Grid", "Rand", "MRate", "SaTE")
DEFAULT_AVERAGE_LCTS = tuple(value / 10.0 for value in range(8, 41, 2))
SWEEP_FLOAT_FIELDS = {
    "average_lcts",
    "realized_average_lcts",
    "throughput_gbps",
    "runtime_ms",
}
PRICE_FLOAT_FIELDS = {
    "realized_average_lcts",
    "selected_route_reachability",
    "front_back_price_mean",
    "side_price_mean",
    "side_to_front_back_price_ratio",
    "throughput_gbps",
    "average_route_hops",
    "price",
}
INT_FIELDS = {
    "seed",
    "front_back_endpoint_count",
    "side_endpoint_count",
    "selected_lisl_index",
    "endpoint_index",
    "lct_index",
}
BOOL_FIELDS = {"possible_graph_connected"}


def parse_float_list(value):
    """Parse a comma-separated list of floating-point values."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    """Parse a comma-separated list of integer values."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _activate_exact_slots(mask, eligible_columns, active_count, rng):
    eligible = np.asarray(
        [
            (satellite, column)
            for satellite in range(mask.shape[0])
            for column in eligible_columns
        ],
        dtype=np.int64,
    )
    if active_count < 0 or active_count > len(eligible):
        raise ValueError("Requested active count exceeds eligible LCT slots")
    chosen = rng.choice(len(eligible), size=active_count, replace=False)
    selected = eligible[chosen]
    mask[selected[:, 0], selected[:, 1]] = 1.0


def build_average_lct_mask(n_sat, average_lcts, seed):
    """Build a reproducible mask with the requested exact constellation mean."""
    if n_sat < 1:
        raise ValueError("n_sat must be positive")
    if average_lcts < 0.0 or average_lcts > 4.0:
        raise ValueError("average_lcts must lie in [0, 4]")
    total = average_lcts * n_sat
    if not np.isclose(total, round(total)):
        raise ValueError("average_lcts * n_sat must be an integer")

    mask = np.zeros((n_sat, 4), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if average_lcts <= 2.0:
        _activate_exact_slots(mask, (0, 1), int(round(total)), rng)
    else:
        mask[:, :2] = 1.0
        extra = int(round((average_lcts - 2.0) * n_sat))
        _activate_exact_slots(mask, (2, 3), extra, rng)
    return mask


def build_exact_three_lct_mask(n_sat, seed):
    """Enable front, back, and exactly one balanced side LCT per satellite."""
    if n_sat < 2 or n_sat % 2:
        raise ValueError("The exact three-LCT test requires an even n_sat")
    mask = np.zeros((n_sat, 4), dtype=np.float32)
    mask[:, :2] = 1.0
    order = np.random.default_rng(seed).permutation(n_sat)
    midpoint = n_sat // 2
    mask[order[:midpoint], 2] = 1.0
    mask[order[midpoint:], 3] = 1.0
    return mask


def build_exact_lct_mask(n_sat, lct_count, seed):
    """Enable an exact two-, three-, or four-LCT layout per satellite."""
    if lct_count == 2:
        mask = np.zeros((n_sat, 4), dtype=np.float32)
        mask[:, :2] = 1.0
        return mask
    if lct_count == 3:
        return build_exact_three_lct_mask(n_sat=n_sat, seed=seed)
    if lct_count == 4:
        return np.ones((n_sat, 4), dtype=np.float32)
    raise ValueError("lct_count must be one of 2, 3, or 4")


def classify_and_average_selected_prices(
    connected_lct,
    forward_prices,
    reverse_prices,
):
    """Average selected-link prices by the LCT class at each endpoint."""
    samples = selected_price_sample_rows(
        connected_lct=connected_lct,
        forward_prices=forward_prices,
        reverse_prices=reverse_prices,
    )
    front_back_prices = [
        row["price"]
        for row in samples
        if row["terminal"] in ("front", "back")
    ]
    side_prices = [
        row["price"] for row in samples if row["terminal"] == "side"
    ]
    if not front_back_prices or not side_prices:
        raise ValueError("Both selected endpoint classes must be non-empty")

    front_back_mean = float(np.mean(front_back_prices))
    side_mean = float(np.mean(side_prices))
    if front_back_mean <= 0.0:
        raise ValueError("Front/back mean price must be positive")
    return {
        "front_back_endpoint_count": len(front_back_prices),
        "side_endpoint_count": len(side_prices),
        "front_back_price_mean": front_back_mean,
        "side_price_mean": side_mean,
        "side_to_front_back_price_ratio": side_mean / front_back_mean,
    }


def selected_price_sample_rows(
    connected_lct,
    forward_prices,
    reverse_prices,
):
    """Return one symmetric congestion-price sample per selected LCT endpoint."""
    connected_lct = np.asarray(connected_lct, dtype=np.int64)
    forward_prices = np.asarray(forward_prices, dtype=np.float64)
    reverse_prices = np.asarray(reverse_prices, dtype=np.float64)
    if connected_lct.ndim != 2 or connected_lct.shape[1] != 2:
        raise ValueError("connected_lct must have shape (n, 2)")
    if len(connected_lct) != len(forward_prices) or len(connected_lct) != len(
        reverse_prices
    ):
        raise ValueError("Price arrays must align with connected_lct")

    link_prices = 0.5 * (forward_prices + reverse_prices)
    terminal_names = ("front", "back", "side", "side")
    rows = []
    for selected_lisl_index, lct_pair in enumerate(connected_lct):
        for endpoint_index, lct_index in enumerate(lct_pair):
            rows.append(
                {
                    "selected_lisl_index": selected_lisl_index,
                    "endpoint_index": endpoint_index,
                    "lct_index": int(lct_index),
                    "terminal": terminal_names[int(lct_index) % 4],
                    "price": float(link_prices[selected_lisl_index]),
                }
            )
    return rows


def write_csv(path, rows):
    """Write a non-empty list of consistently shaped dictionaries."""
    if not rows:
        raise ValueError("Cannot write an empty result table")
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    """Read an LCT experiment table and restore numeric field types."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as fp:
        for raw in csv.DictReader(fp):
            row = dict(raw)
            for field in SWEEP_FLOAT_FIELDS | PRICE_FLOAT_FIELDS:
                if field in row:
                    row[field] = float(row[field])
            for field in INT_FIELDS:
                if field in row:
                    row[field] = int(row[field])
            for field in BOOL_FIELDS:
                if field in row:
                    row[field] = row[field].lower() == "true"
            rows.append(row)
    return rows


def _mean_and_std(rows, metric):
    values = np.asarray([row[metric] for row in rows], dtype=np.float64)
    return float(np.mean(values)), (
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    )


def validate_sweep_rows(rows, levels, seeds):
    """Validate completeness and basic bounds of a full sweep result matrix."""
    expected = len(levels) * len(seeds) * len(METHODS)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} sweep rows, found {len(rows)}")
    keys = set()
    for row in rows:
        key = (row["average_lcts"], row["seed"], row["method"])
        if key in keys:
            raise ValueError(f"Duplicate sweep key {key}")
        keys.add(key)
        if row["throughput_gbps"] < 0.0 or row["runtime_ms"] < 0.0:
            raise ValueError(f"Negative sweep metric for {key}")
        if not np.isclose(
            row["realized_average_lcts"],
            row["average_lcts"],
        ):
            raise ValueError(f"Incorrect realized LCT count for {key}")
    for level in levels:
        for seed in seeds:
            methods = {
                row["method"]
                for row in rows
                if row["average_lcts"] == level and row["seed"] == seed
            }
            if methods != set(METHODS):
                raise ValueError(
                    f"Method mismatch for average_lcts={level}, seed={seed}"
                )


def summarize_sweep_rows(rows):
    """Aggregate sweep results by LCT level and method."""
    summary = []
    levels = sorted({row["average_lcts"] for row in rows})
    for level in levels:
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["average_lcts"] == level and row["method"] == method
            ]
            throughput_mean, throughput_std = _mean_and_std(
                selected, "throughput_gbps"
            )
            runtime_mean, runtime_std = _mean_and_std(selected, "runtime_ms")
            summary.append(
                {
                    "average_lcts": level,
                    "method": method,
                    "n_cases": len(selected),
                    "throughput_gbps_mean": throughput_mean,
                    "throughput_gbps_std": throughput_std,
                    "runtime_ms_mean": runtime_mean,
                    "runtime_ms_std": runtime_std,
                }
            )
    return summary


def validate_price_rows(rows, seeds):
    """Validate the exact three-LCT price result matrix."""
    if len(rows) != len(seeds):
        raise ValueError(f"Expected {len(seeds)} price rows, found {len(rows)}")
    if {row["seed"] for row in rows} != set(seeds):
        raise ValueError("Price-result seeds do not match the requested seeds")
    for row in rows:
        if not np.isclose(row["realized_average_lcts"], 3.0):
            raise ValueError("The exact three-LCT mask does not average to three")
        if not row["possible_graph_connected"]:
            raise ValueError(f"Disconnected possible graph for seed {row['seed']}")
        if (
            row["front_back_endpoint_count"] < 1
            or row["side_endpoint_count"] < 1
        ):
            raise ValueError(
                f"Missing selected endpoint class for seed {row['seed']}"
            )
        if row["selected_route_reachability"] <= 0.0:
            raise ValueError(f"No reachable selected routes for seed {row['seed']}")


def validate_price_sample_rows(sample_rows, result_rows):
    """Check terminal-level price samples against realization aggregates."""
    if not sample_rows:
        raise ValueError("The terminal-level price sample table is empty")
    result_by_seed = {row["seed"]: row for row in result_rows}
    if len(result_by_seed) != len(result_rows):
        raise ValueError("Duplicate seed in aggregate price results")
    if {row["seed"] for row in sample_rows} != set(result_by_seed):
        raise ValueError("Price-sample seeds do not match aggregate results")

    seen = set()
    for sample in sample_rows:
        key = (
            sample["seed"],
            sample["selected_lisl_index"],
            sample["endpoint_index"],
        )
        if key in seen:
            raise ValueError(f"Duplicate terminal-level price sample {key}")
        seen.add(key)
        if sample["endpoint_index"] not in (0, 1):
            raise ValueError(f"Invalid selected-LISL endpoint index for {key}")
        if sample["terminal"] not in ("front", "back", "side"):
            raise ValueError(f"Unknown terminal class for {key}")
        if sample["price"] < 0.0:
            raise ValueError(f"Negative congestion price for {key}")

    for seed, result in result_by_seed.items():
        selected = [row for row in sample_rows if row["seed"] == seed]
        terminals = {row["terminal"] for row in selected}
        if terminals != {"front", "back", "side"}:
            raise ValueError(f"Missing terminal class for seed {seed}")
        front_back = [
            row["price"]
            for row in selected
            if row["terminal"] in ("front", "back")
        ]
        side = [
            row["price"] for row in selected if row["terminal"] == "side"
        ]
        if len(front_back) != result["front_back_endpoint_count"]:
            raise ValueError(f"Front/back sample-count mismatch for seed {seed}")
        if len(side) != result["side_endpoint_count"]:
            raise ValueError(f"Side sample-count mismatch for seed {seed}")
        if not np.isclose(
            np.mean(front_back),
            result["front_back_price_mean"],
        ):
            raise ValueError(f"Front/back price-mean mismatch for seed {seed}")
        if not np.isclose(np.mean(side), result["side_price_mean"]):
            raise ValueError(f"Side price-mean mismatch for seed {seed}")


def summarize_price_rows(rows):
    """Aggregate exact three-LCT price metrics across realizations."""
    metrics = (
        "front_back_price_mean",
        "side_price_mean",
        "side_to_front_back_price_ratio",
        "throughput_gbps",
        "average_route_hops",
        "selected_route_reachability",
    )
    summary = {"n_cases": len(rows)}
    for metric in metrics:
        mean, std = _mean_and_std(rows, metric)
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
    return summary
