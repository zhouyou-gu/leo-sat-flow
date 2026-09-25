import unittest
import numpy as np
from flow_delay_metrics import route_delays, weighted_stats, EARTH_RADIUS_KM, LIGHT_SPEED_KM_S

class DelayTests(unittest.TestCase):
    def test_distances_and_exclusions(self):
        positions = np.array([[0,0,0],[1,0,0],[1,1,0]], dtype=float) / EARTH_RADIUS_KM
        paths = np.array([[0,1,-1],[0,1,2],[0,1,-1],[-1,-1,-1]])
        d = route_delays(positions, paths, [2,3,2,0], [1.,2.,0.,0.], [0]*4, [1,2,1,2], [[0,1],[1,2]])
        np.testing.assert_allclose(d[:2], np.array([1,2])/LIGHT_SPEED_KM_S*1000)
        self.assertTrue(np.isnan(d[2:]).all())
    def test_weighted_statistics(self):
        s = weighted_stats(np.array([1.,10.,np.nan]), np.array([96.,4.,0.]))
        self.assertAlmostEqual(s['mean_delay_ms'], 1.36)
        self.assertEqual(s['p95_delay_ms'], 1.)
        self.assertIsNone(weighted_stats(np.array([np.nan]), np.array([0.]))['mean_delay_ms'])
    def test_invalid_routes(self):
        args = (np.zeros((2,3)), np.array([[0,1]]), [2], [1.], [0], [1])
        with self.assertRaises(AssertionError): route_delays(*args, [])
        with self.assertRaises(AssertionError): route_delays(args[0], args[1], [0], [1.], [0], [1], [[0,1]])

if __name__ == '__main__': unittest.main()
