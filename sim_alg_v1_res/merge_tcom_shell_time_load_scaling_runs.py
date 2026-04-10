#!/usr/bin/env python3

import argparse
import json
import os
from datetime import datetime

import pandas as pd


def list_candidate_runs(parent_dir):
    candidates = []
    for entry in os.listdir(parent_dir):
        run_dir = os.path.join(parent_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        if not os.path.isfile(os.path.join(run_dir, "results.csv")):
            continue
        if not os.path.isfile(os.path.join(run_dir, "shell_metadata.json")):
            continue
        candidates.append(run_dir)
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates


def load_metadata(run_dir):
    metadata_path = os.path.join(run_dir, "shell_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent-dir",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "test_tcom_shell_time_load_scaling",
        ),
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        help="Explicit run directories to merge. If omitted, the latest run for each active-user percentage is used.",
    )
    args = parser.parse_args()

    if args.inputs:
        input_dirs = [os.path.abspath(path) for path in args.inputs]
    else:
        input_dirs = []
        seen = set()
        for run_dir in list_candidate_runs(args.parent_dir):
            metadata = load_metadata(run_dir)
            percentages = metadata.get("active_user_percentages", [])
            if len(percentages) != 1:
                continue
            percentage = float(percentages[0])
            if percentage in seen:
                continue
            seen.add(percentage)
            input_dirs.append(run_dir)
            if len(seen) == 5:
                break
        input_dirs.sort()

    if not input_dirs:
        raise FileNotFoundError("No load-scaling run directories were found to merge.")

    dataframes = []
    metadata_list = []
    for run_dir in input_dirs:
        dataframes.append(pd.read_csv(os.path.join(run_dir, "results.csv")))
        metadata_list.append(load_metadata(run_dir))

    merged = pd.concat(dataframes, ignore_index=True)
    merged = merged.sort_values(
        by=["active_user_percentage", "offset_minutes", "method"],
        kind="stable",
    )

    timestamp = datetime.now().strftime("%Y-%B-%d-%H-%M-%S")
    output_dir = os.path.join(args.parent_dir, f"merged-{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    merged.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    base_metadata = dict(metadata_list[0])
    percentages = sorted(
        {
            float(percentage)
            for metadata in metadata_list
            for percentage in metadata.get("active_user_percentages", [])
        }
    )
    base_metadata["active_user_percentages"] = percentages
    base_metadata["merged_inputs"] = input_dirs
    with open(os.path.join(output_dir, "shell_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(base_metadata, fp, indent=2)

    print(output_dir)


if __name__ == "__main__":
    main()
