import unittest

from sim_alg_j1_res import test_ld_starlink_1000_exact_lct_prices as experiment


class ExactLCTPriceExperimentTest(unittest.TestCase):
    def test_default_cases_and_field_of_regard(self):
        self.assertEqual(experiment.DEFAULT_LCT_COUNTS, (2, 3, 4))
        self.assertEqual(experiment.DEFAULT_FOR_THETA_HALF, 60.0)

    def test_result_row_records_exact_lct_count_and_metrics(self):
        row = experiment.build_result_row(
            lct_count=3,
            seed=7,
            possible_graph_connected=True,
            lengths=[3, 4, 0],
            prices=[0.1, 0.2, 0.3, 0.4],
            primal_objective=-123.5,
        )
        self.assertEqual(row["lct_count"], 3)
        self.assertEqual(row["seed"], 7)
        self.assertAlmostEqual(row["realized_average_lcts"], 3.0)
        self.assertAlmostEqual(row["selected_route_reachability"], 2.0 / 3.0)
        self.assertEqual(row["selected_endpoint_count"], 4)
        self.assertAlmostEqual(row["congestion_price_mean"], 0.25)
        self.assertAlmostEqual(row["throughput_gbps"], 123.5)


if __name__ == "__main__":
    unittest.main()
