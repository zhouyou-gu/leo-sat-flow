import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim_alg_j1_res.lct_experiment_helpers import (
    DEFAULT_AVERAGE_LCTS,
    METHODS,
    build_average_lct_mask,
    build_exact_lct_mask,
    build_exact_three_lct_mask,
    classify_and_average_selected_prices,
    read_csv,
    summarize_price_rows,
    summarize_sweep_rows,
    validate_price_rows,
    validate_sweep_rows,
    write_csv,
)
from sim_alg_j1_res.test_ld_starlink_1000_varying_n_lct_extended import (
    build_result_row,
    heuristic_arguments,
)
from sim_alg_j1_res import lct_experiment_helpers as lct_helpers
from sim_alg_j1_res import test_ld_starlink_1000_three_lct_prices as three_lct


build_price_result_row = three_lct.build_price_result_row


class LCTExperimentHelpersTest(unittest.TestCase):
    def test_exact_lct_masks_enable_expected_terminal_sets(self):
        mask_two = build_exact_lct_mask(n_sat=8, lct_count=2, seed=9)
        mask_three = build_exact_lct_mask(n_sat=8, lct_count=3, seed=9)
        mask_four = build_exact_lct_mask(n_sat=8, lct_count=4, seed=9)

        np.testing.assert_allclose(mask_two[:, :2], 1.0)
        np.testing.assert_allclose(mask_two[:, 2:], 0.0)
        np.testing.assert_allclose(mask_three[:, :2], 1.0)
        np.testing.assert_allclose(mask_three.sum(axis=1), 3.0)
        self.assertEqual(int(mask_three[:, 2].sum()), 4)
        self.assertEqual(int(mask_three[:, 3].sum()), 4)
        np.testing.assert_allclose(mask_four, 1.0)

    def test_exact_lct_mask_rejects_unsupported_counts(self):
        for lct_count in (1, 5):
            with self.subTest(lct_count=lct_count):
                with self.assertRaises(ValueError):
                    build_exact_lct_mask(
                        n_sat=8,
                        lct_count=lct_count,
                        seed=0,
                    )

    def test_default_sweep_runs_from_point_eight_through_four(self):
        self.assertEqual(DEFAULT_AVERAGE_LCTS[0], 0.8)
        self.assertEqual(DEFAULT_AVERAGE_LCTS[-1], 4.0)
        self.assertEqual(len(DEFAULT_AVERAGE_LCTS), 17)
        self.assertTrue(np.allclose(np.diff(DEFAULT_AVERAGE_LCTS), 0.2))

    def test_average_mask_has_exact_requested_count(self):
        for average in (0.8, 1.4, 2.0, 2.2, 3.0, 3.8, 4.0):
            mask = build_average_lct_mask(1000, average, seed=7)
            self.assertEqual(mask.shape, (1000, 4))
            self.assertTrue(np.isin(mask, (0.0, 1.0)).all())
            self.assertAlmostEqual(float(mask.sum()) / 1000.0, average)
            if average <= 2.0:
                self.assertTrue((mask[:, 2:] == 0.0).all())
            else:
                self.assertTrue((mask[:, :2] == 1.0).all())

    def test_average_mask_is_seed_reproducible(self):
        first = build_average_lct_mask(1000, 2.6, seed=11)
        second = build_average_lct_mask(1000, 2.6, seed=11)
        third = build_average_lct_mask(1000, 2.6, seed=12)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))

    def test_exact_three_mask_has_two_longitudinal_and_one_side_lct(self):
        mask = build_exact_three_lct_mask(1000, seed=3)
        np.testing.assert_array_equal(mask[:, :2], np.ones((1000, 2)))
        np.testing.assert_array_equal(mask.sum(axis=1), np.full(1000, 3.0))
        np.testing.assert_array_equal(mask[:, 2:].sum(axis=1), np.ones(1000))
        self.assertEqual(int(mask[:, 2].sum()), 500)
        self.assertEqual(int(mask[:, 3].sum()), 500)

    def test_selected_prices_are_grouped_by_endpoint_terminal_types(self):
        connected_lct = np.asarray(
            [
                [0, 5],
                [4, 9],
                [2, 7],
                [6, 11],
            ],
            dtype=np.int64,
        )
        stats = classify_and_average_selected_prices(
            connected_lct=connected_lct,
            forward_prices=np.asarray([1.0, 3.0, 5.0, 7.0]),
            reverse_prices=np.asarray([3.0, 5.0, 7.0, 9.0]),
        )
        self.assertEqual(stats["front_back_endpoint_count"], 4)
        self.assertEqual(stats["side_endpoint_count"], 4)
        self.assertAlmostEqual(stats["front_back_price_mean"], 3.0)
        self.assertAlmostEqual(stats["side_price_mean"], 7.0)
        self.assertAlmostEqual(
            stats["side_to_front_back_price_ratio"],
            7.0 / 3.0,
        )

    def test_mixed_terminal_pairs_contribute_one_endpoint_to_each_class(self):
        stats = classify_and_average_selected_prices(
            connected_lct=np.asarray([[0, 6], [2, 5]], dtype=np.int64),
            forward_prices=np.asarray([1.0, 5.0]),
            reverse_prices=np.asarray([3.0, 7.0]),
        )
        self.assertEqual(stats["front_back_endpoint_count"], 2)
        self.assertEqual(stats["side_endpoint_count"], 2)
        self.assertAlmostEqual(stats["front_back_price_mean"], 4.0)
        self.assertAlmostEqual(stats["side_price_mean"], 4.0)

    def test_selected_price_samples_keep_front_back_and_side_separate(self):
        self.assertTrue(
            hasattr(lct_helpers, "selected_price_sample_rows"),
            "selected_price_sample_rows must export terminal-level prices",
        )
        rows = lct_helpers.selected_price_sample_rows(
            connected_lct=np.asarray(
                [[0, 5], [2, 7], [4, 11]],
                dtype=np.int64,
            ),
            forward_prices=np.asarray([1.0, 5.0, 9.0]),
            reverse_prices=np.asarray([3.0, 7.0, 11.0]),
        )
        self.assertEqual(
            [row["terminal"] for row in rows],
            ["front", "back", "side", "side", "front", "side"],
        )
        np.testing.assert_allclose(
            [row["price"] for row in rows],
            [2.0, 2.0, 6.0, 6.0, 10.0, 10.0],
        )
        self.assertEqual(
            [row["lct_index"] for row in rows],
            [0, 5, 2, 7, 4, 11],
        )

    def test_price_sample_table_matches_aggregate_result_rows(self):
        self.assertTrue(
            hasattr(lct_helpers, "validate_price_sample_rows"),
            "Price samples must be checked against aggregate result rows",
        )
        samples = [
            {
                "seed": 0,
                "selected_lisl_index": index // 2,
                "endpoint_index": index % 2,
                "lct_index": index,
                "terminal": terminal,
                "price": price,
            }
            for index, (terminal, price) in enumerate(
                (
                    ("front", 2.0),
                    ("back", 4.0),
                    ("side", 3.0),
                    ("side", 5.0),
                )
            )
        ]
        result_rows = [
            {
                "seed": 0,
                "front_back_endpoint_count": 2,
                "side_endpoint_count": 2,
                "front_back_price_mean": 3.0,
                "side_price_mean": 4.0,
            }
        ]
        lct_helpers.validate_price_sample_rows(samples, result_rows)
        inconsistent = [dict(result_rows[0], side_price_mean=4.5)]
        with self.assertRaises(ValueError):
            lct_helpers.validate_price_sample_rows(samples, inconsistent)

    def test_sweep_and_price_tables_validate_and_summarize(self):
        sweep_rows = []
        for average in (0.8, 1.0):
            for seed in (0, 1):
                for method_index, method in enumerate(METHODS):
                    sweep_rows.append(
                        {
                            "average_lcts": average,
                            "seed": seed,
                            "method": method,
                            "realized_average_lcts": average,
                            "throughput_gbps": 10.0 + method_index,
                            "runtime_ms": 1.0 + method_index,
                            "possible_graph_connected": True,
                        }
                    )
        validate_sweep_rows(sweep_rows, [0.8, 1.0], [0, 1])
        self.assertEqual(len(summarize_sweep_rows(sweep_rows)), 10)

        price_rows = [
            {
                "seed": seed,
                "realized_average_lcts": 3.0,
                "possible_graph_connected": True,
                "selected_route_reachability": 1.0,
                "front_back_endpoint_count": 100,
                "side_endpoint_count": 50,
                "front_back_price_mean": 2.0 + seed,
                "side_price_mean": 4.0 + seed,
                "side_to_front_back_price_ratio": (4.0 + seed) / (2.0 + seed),
                "throughput_gbps": 100.0,
                "average_route_hops": 5.0,
            }
            for seed in (0, 1)
        ]
        validate_price_rows(price_rows, [0, 1])
        summary = summarize_price_rows(price_rows)
        self.assertEqual(summary["n_cases"], 2)
        self.assertAlmostEqual(summary["front_back_price_mean_mean"], 2.5)
        self.assertAlmostEqual(summary["side_price_mean_mean"], 4.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rows.csv"
            write_csv(path, price_rows)
            restored = read_csv(path)
        self.assertEqual(restored, price_rows)

    def test_sweep_driver_mapping_and_result_row(self):
        self.assertEqual(heuristic_arguments("+Grid", 7), ("grid", "ospf", 7))
        self.assertEqual(heuristic_arguments("Rand", 7), ("rand", "ospf", 7))
        self.assertEqual(heuristic_arguments("MRate", 7), ("mwm", "ospf", 7))
        self.assertEqual(heuristic_arguments("SaTE", 7), ("mwm", "spf", 7))
        row = build_result_row(
            average_lcts=3.2,
            seed=4,
            method="DeepLaDu",
            realized_average_lcts=3.2,
            primal_objective=-123.0,
            runtime_ms=8.5,
            possible_graph_connected=True,
        )
        self.assertEqual(row["throughput_gbps"], 123.0)

    def test_exact_three_lct_result_row(self):
        row = build_price_result_row(
            seed=2,
            possible_graph_connected=True,
            lengths=np.asarray([4, 6, 0]),
            price_stats={
                "front_back_endpoint_count": 8,
                "side_endpoint_count": 4,
                "front_back_price_mean": 2.0,
                "side_price_mean": 5.0,
                "side_to_front_back_price_ratio": 2.5,
            },
            primal_objective=-90.0,
        )
        self.assertEqual(row["realized_average_lcts"], 3.0)
        self.assertAlmostEqual(row["selected_route_reachability"], 2.0 / 3.0)
        self.assertAlmostEqual(row["average_route_hops"], 4.0)
        self.assertEqual(row["throughput_gbps"], 90.0)

    def test_walker_delta_selector_uses_fully_regular_constellation(self):
        self.assertTrue(hasattr(three_lct, "constellation_ratio"))
        self.assertEqual(three_lct.constellation_ratio("starlink"), 0.0)
        self.assertEqual(three_lct.constellation_ratio("walker-delta"), 1.0)
        with self.assertRaises(ValueError):
            three_lct.constellation_ratio("unknown")

    def test_three_lct_field_of_regard_uses_45_degree_half_angle(self):
        self.assertTrue(
            hasattr(three_lct, "configure_field_of_regard"),
            "The three-LCT driver must configure its field of regard",
        )
        simulation = type("SimulationStub", (), {})()
        three_lct.configure_field_of_regard(simulation, theta_half=45.0)
        self.assertEqual(simulation.FOR_THETA_HALF, 45.0)
        self.assertAlmostEqual(simulation.cos_threshold, np.sqrt(0.5))


if __name__ == "__main__":
    unittest.main()
