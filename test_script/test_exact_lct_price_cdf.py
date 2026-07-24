import csv
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_NAME = "sim_alg_j1_res.figure.plot_exact_lct_price_cdf"


def load_plot_module(test_case):
    test_case.assertIsNotNone(
        importlib.util.find_spec(MODULE_NAME),
        "The exact-LCT price CDF plotting module must exist",
    )
    return importlib.import_module(MODULE_NAME)


class ExactLCTPriceCDFTest(unittest.TestCase):
    def write_samples(self, path):
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=("lct_count", "seed", "terminal", "price"),
            )
            writer.writeheader()
            for lct_count in (2, 3, 4):
                writer.writerows(
                    [
                        {
                            "lct_count": lct_count,
                            "seed": 0,
                            "terminal": "front",
                            "price": 0.01 * lct_count,
                        },
                        {
                            "lct_count": lct_count,
                            "seed": 0,
                            "terminal": "back",
                            "price": 0.02 * lct_count,
                        },
                    ]
                )

    def test_empirical_cdf_sorts_values_and_reaches_one(self):
        plot_module = load_plot_module(self)
        values, probabilities = plot_module.empirical_cdf([3.0, 1.0, 2.0])
        np.testing.assert_allclose(values, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(probabilities, [1.0 / 3.0, 2.0 / 3.0, 1.0])

    def test_loader_groups_all_endpoint_prices_by_lct_count(self):
        plot_module = load_plot_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            samples_path = Path(tmp_dir) / "price_samples.csv"
            self.write_samples(samples_path)
            prices = plot_module.load_lct_count_prices(samples_path)
        self.assertEqual(set(prices), {2, 3, 4})
        np.testing.assert_allclose(prices[2], [0.02, 0.04])
        np.testing.assert_allclose(prices[3], [0.03, 0.06])
        np.testing.assert_allclose(prices[4], [0.04, 0.08])

    def test_figure_contains_three_labeled_cdf_lines(self):
        plot_module = load_plot_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            samples_path = Path(tmp_dir) / "price_samples.csv"
            self.write_samples(samples_path)
            figure = plot_module.make_figure(samples_path)
        axis = figure.axes[0]
        self.assertEqual(
            [line.get_label() for line in axis.lines],
            ["2 LCTs", "3 LCTs", "4 LCTs"],
        )
        self.assertEqual(axis.get_xlabel(), "Congestion Price")
        self.assertEqual(axis.get_ylabel(), "Empirical CDF")

    def test_cdf_lines_use_spaced_hollow_markers(self):
        plot_module = load_plot_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            samples_path = Path(tmp_dir) / "price_samples.csv"
            self.write_samples(samples_path)
            figure = plot_module.make_figure(samples_path)
        lines = figure.axes[0].lines
        self.assertEqual([line.get_marker() for line in lines], ["o", "s", "^"])
        self.assertEqual(
            [line.get_markevery() for line in lines],
            [(0.00, 0.12), (0.04, 0.12), (0.08, 0.12)],
        )
        self.assertTrue(
            all(line.get_markerfacecolor() == "none" for line in lines)
        )


if __name__ == "__main__":
    unittest.main()
