import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim_alg_j1_res.figure.plot_test_ld_starlink_1000_constellation_varying_lct_for import (
    METHOD_NAMES,
    load_lct_summary,
)


class LCTFigureDataTest(unittest.TestCase):
    def test_summary_loader_returns_all_levels_in_plot_method_order(self):
        levels = np.round(np.arange(0.8, 4.0 + 0.1, 0.2), 1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=[
                        "average_lcts",
                        "method",
                        "n_cases",
                        "throughput_gbps_mean",
                    ],
                )
                writer.writeheader()
                for level_index, level in enumerate(levels):
                    for method_index, method in enumerate(METHOD_NAMES):
                        writer.writerow(
                            {
                                "average_lcts": level,
                                "method": method,
                                "n_cases": 20,
                                "throughput_gbps_mean": 10 * level_index
                                + method_index,
                            }
                        )

            loaded_levels, throughput = load_lct_summary(summary_path)

        np.testing.assert_allclose(loaded_levels, levels)
        self.assertEqual(throughput.shape, (17, 5))
        np.testing.assert_allclose(throughput[0], np.arange(5))
        np.testing.assert_allclose(throughput[-1], 160 + np.arange(5))


if __name__ == "__main__":
    unittest.main()
