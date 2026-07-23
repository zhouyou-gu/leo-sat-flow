import os
import tempfile
import unittest

import numpy as np

import sim_alg_j1_res.varying_load_helpers as varying_load_helpers
from sim_alg_j1_res.varying_load_helpers import (
    METHODS,
    parse_float_list,
    parse_int_list,
    read_csv,
    summarize_rows,
    validate_rows,
    write_csv,
)
from sim_alg_j1_res.test_ld_starlink_1000_varying_load import (
    DEFAULT_ACTIVE_USER_PERCENTAGES,
    build_result_row,
    heuristic_arguments,
)


class VaryingLoadHelpersTest(unittest.TestCase):
    def make_rows(self):
        rows = []
        for load in (1e-4, 2e-4):
            for seed in (0, 1):
                for method_index, method in enumerate(METHODS):
                    offered = 100.0 * (load / 1e-4)
                    served = offered * (0.2 + 0.05 * method_index)
                    rows.append(
                        {
                            "active_user_percentage": load,
                            "seed": seed,
                            "method": method,
                            "offered_demand_gbps": offered,
                            "served_throughput_gbps": served,
                            "served_ratio": served / offered,
                            "runtime_ms": 10.0 + method_index,
                            "flow_pair_count": 50,
                            "demand_satellite_count": 10,
                            "matched_satellite_pair_count": 20,
                            "allocated_flow_count": 30,
                        }
                    )
        return rows

    def test_parse_lists(self):
        self.assertEqual(parse_float_list("1e-4, 2e-4"), [1e-4, 2e-4])
        self.assertEqual(parse_int_list("0, 2,4"), [0, 2, 4])

    def test_validate_and_summarize_complete_matrix(self):
        rows = self.make_rows()
        validate_rows(rows, [1e-4, 2e-4], [0, 1])
        summary = summarize_rows(rows, [1e-4, 2e-4])
        self.assertEqual(len(summary), 2 * len(METHODS))
        deep = next(
            row
            for row in summary
            if row["active_user_percentage"] == 1e-4
            and row["method"] == "DeepLaDu"
        )
        self.assertEqual(deep["n_cases"], 2)
        self.assertAlmostEqual(deep["offered_demand_gbps_mean"], 100.0)
        self.assertAlmostEqual(deep["served_ratio_mean"], 0.2)

    def test_validate_rejects_missing_method_case(self):
        rows = self.make_rows()[:-1]
        with self.assertRaisesRegex(ValueError, "Expected 20 rows, found 19"):
            validate_rows(rows, [1e-4, 2e-4], [0, 1])

    def test_validate_rejects_out_of_range_ratio(self):
        rows = self.make_rows()
        rows[0]["served_ratio"] = 1.01
        with self.assertRaisesRegex(ValueError, "served ratio"):
            validate_rows(rows, [1e-4, 2e-4], [0, 1])

    def test_csv_round_trip_preserves_field_types(self):
        rows = self.make_rows()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "results.csv")
            write_csv(path, rows)
            restored = read_csv(path)
        self.assertEqual(restored, rows)

    def test_default_grid_and_heuristic_mapping(self):
        self.assertEqual(
            DEFAULT_ACTIVE_USER_PERCENTAGES,
            [5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
        )
        self.assertEqual(heuristic_arguments("MRate", 7), ("mwm", "ospf", 7))
        self.assertEqual(heuristic_arguments("+Grid", 7), ("grid", "ospf", 7))
        self.assertEqual(heuristic_arguments("Rand", 7), ("rand", "ospf", 7))
        self.assertEqual(heuristic_arguments("SaTE", 7), ("mwm", "spf", 7))

    def test_build_result_row_computes_served_ratio_and_counts(self):
        row = build_result_row(
            active_user_percentage=1e-4,
            seed=3,
            method="DeepLaDu",
            offered_demand_gbps=200.0,
            primal_objective=-50.0,
            runtime_ms=12.5,
            flow_pair_count=100,
            demand_satellite_count=20,
            matched_satellite_pair_count=300,
            rates=np.asarray([0.0, 4.0, 5.0]),
        )
        self.assertEqual(row["served_throughput_gbps"], 50.0)
        self.assertEqual(row["served_ratio"], 0.25)
        self.assertEqual(row["allocated_flow_count"], 2)

    def test_traffic_demand_rate_series_collapses_method_duplicates(self):
        summary = summarize_rows(self.make_rows(), [1e-4, 2e-4])
        self.assertEqual(
            varying_load_helpers.traffic_demand_rate_series(summary),
            [(1e-4, 100.0), (2e-4, 200.0)],
        )

    def test_demand_satisfaction_ratio_is_reported_as_percentage(self):
        self.assertAlmostEqual(
            varying_load_helpers.demand_satisfaction_percentage(0.357),
            35.7,
        )

    def test_per_user_throughput_converts_ratio_to_gbps(self):
        self.assertTrue(
            hasattr(varying_load_helpers, "per_user_throughput_gbps"),
            "per_user_throughput_gbps must be defined",
        )
        self.assertAlmostEqual(
            varying_load_helpers.per_user_throughput_gbps(
                served_ratio=0.25,
                per_user_demand_gbps=0.1,
            ),
            0.025,
        )

    def test_per_user_throughput_converts_ratio_to_mbps(self):
        self.assertTrue(
            hasattr(varying_load_helpers, "per_user_throughput_mbps"),
            "per_user_throughput_mbps must be defined",
        )
        self.assertAlmostEqual(
            varying_load_helpers.per_user_throughput_mbps(
                served_ratio=0.25,
                per_user_demand_gbps=0.1,
            ),
            25.0,
        )

if __name__ == "__main__":
    unittest.main()
