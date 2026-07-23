"""Pure result-table helpers for the DeepLaDu traffic-load sweep."""

import csv
import numpy as np


METHODS = ("DeepLaDu", "MRate", "+Grid", "Rand", "SaTE")
METRICS = (
    "offered_demand_gbps",
    "served_throughput_gbps",
    "served_ratio",
    "runtime_ms",
    "flow_pair_count",
    "demand_satellite_count",
    "matched_satellite_pair_count",
    "allocated_flow_count",
)


def parse_float_list(value):
    """Parse a comma-separated list of floats."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    """Parse a comma-separated list of integers."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def per_user_throughput_gbps(served_ratio, per_user_demand_gbps):
    """Convert a served-demand ratio into average per-user throughput in Gbps."""
    return float(served_ratio) * float(per_user_demand_gbps)


def per_user_throughput_mbps(served_ratio, per_user_demand_gbps):
    """Convert a served-demand ratio into average per-user throughput in Mbps."""
    return per_user_throughput_gbps(served_ratio, per_user_demand_gbps) * 1000.0


def traffic_demand_rate_series(summary_rows):
    """Return one method-independent aggregate demand rate per traffic load."""
    by_load = {}
    for row in summary_rows:
        load = float(row["active_user_percentage"])
        by_load.setdefault(load, []).append(
            float(row["offered_demand_gbps_mean"])
        )
    series = []
    for load, values in sorted(by_load.items()):
        if not np.allclose(values, values[0]):
            raise ValueError(
                f"Traffic demand rate differs across methods at load {load}"
            )
        series.append((load, values[0]))
    return series


def demand_satisfaction_percentage(served_ratio):
    """Convert a network-throughput-to-demand ratio to a percentage."""
    return float(served_ratio) * 100.0


def write_csv(path, rows):
    """Write a non-empty list of consistently shaped dictionaries."""
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    """Read case-level results and restore numeric field types."""
    float_fields = {
        "active_user_percentage",
        "offered_demand_gbps",
        "served_throughput_gbps",
        "served_ratio",
        "runtime_ms",
    }
    int_fields = {
        "seed",
        "flow_pair_count",
        "demand_satellite_count",
        "matched_satellite_pair_count",
        "allocated_flow_count",
    }
    rows = []
    with open(path, newline="", encoding="utf-8") as fp:
        for raw in csv.DictReader(fp):
            row = dict(raw)
            for field in float_fields:
                row[field] = float(row[field])
            for field in int_fields:
                row[field] = int(row[field])
            rows.append(row)
    return rows


def validate_rows(rows, loads, seeds):
    """Validate completeness and physical bounds of case-level results."""
    expected_count = len(loads) * len(seeds) * len(METHODS)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")

    expected_methods = set(METHODS)
    seen = set()
    for row in rows:
        key = (row["active_user_percentage"], row["seed"], row["method"])
        if key in seen:
            raise ValueError(f"Duplicate result key: {key}")
        seen.add(key)

        ratio = row["served_ratio"]
        if ratio < -1e-9 or ratio > 1.0 + 1e-9:
            raise ValueError(f"Invalid served ratio {ratio} for {key}")
        if row["served_throughput_gbps"] < -1e-9:
            raise ValueError(f"Negative throughput for {key}")
        if row["served_throughput_gbps"] > row["offered_demand_gbps"] + 1e-6:
            raise ValueError(f"Served throughput exceeds offered demand for {key}")

    for load in loads:
        for seed in seeds:
            methods = {
                row["method"]
                for row in rows
                if row["active_user_percentage"] == load and row["seed"] == seed
            }
            if methods != expected_methods:
                raise ValueError(
                    f"Method mismatch for load={load}, seed={seed}: {sorted(methods)}"
                )


def summarize_rows(rows, loads):
    """Aggregate mean and sample standard deviation by load and method."""
    summary_rows = []
    for load in loads:
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["active_user_percentage"] == load
                and row["method"] == method
            ]
            summary = {
                "active_user_percentage": load,
                "method": method,
                "n_cases": len(selected),
            }
            for metric in METRICS:
                values = np.asarray(
                    [row[metric] for row in selected], dtype=np.float64
                )
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
            summary_rows.append(summary)
    return summary_rows
