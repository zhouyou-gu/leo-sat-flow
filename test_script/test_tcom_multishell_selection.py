"""Regression checks for the opt-in Comment 2.1 constellation selection."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_mld.tle import load_tle_valid_at_times, select_shifted_shell_block


def satellite(satnum, raan, inc=53.22, motion=15.087):
    return SimpleNamespace(model=SimpleNamespace(satnum=satnum, inclo=math.radians(inc),
        nodeo=math.radians(raan), no_kozai=motion*2*math.pi/1440))


class ShellSelectionTests(unittest.TestCase):
    def test_minimum_block_can_cross_zero_raan(self):
        sats = [satellite(i,r) for i,r in enumerate([358,359,1,90,180,270])]
        idx, meta = select_shifted_shell_block(sats,53.22,15.087,3,0)
        self.assertEqual(set(idx),{0,1,2})
        self.assertAlmostEqual(meta['raan_span_deg'],3)

    def test_shifted_blocks_keep_membership_and_population(self):
        sats = [satellite(i,i*30) for i in range(12)] + [satellite(99,0,43,15.024)]
        blocks = [select_shifted_shell_block(sats,53.22,15.087,5,k)[0] for k in range(3)]
        self.assertTrue(all(len(set(b)) == 5 and 12 not in b for b in blocks))
        self.assertEqual(len(set().union(*(set(b) for b in blocks))),12)
        with self.assertRaises(ValueError):
            select_shifted_shell_block(sats,43,15.024,2,0)

    def test_loader_rejects_errors_at_any_requested_epoch(self):
        sats = [satellite(i,0) for i in range(4)]
        positions = np.ones((4,2,3)); velocities = positions.copy()
        errors = np.zeros((4,2), dtype=int)
        errors[1,1] = 6
        velocities[2,0,0] = np.nan
        positions[3,1,0] = np.inf
        fake_array = SimpleNamespace(sgp4=lambda jd,fr:(errors,positions,velocities))
        fake_ts = object()  # No now() available: filtering must use explicit times.
        dates = [SimpleNamespace(utc=(2025,7,16,16,0,0)),SimpleNamespace(utc=(2025,7,16,19,0,0))]
        with patch('sim_mld.tle.load.timescale',return_value=fake_ts), \
             patch('sim_mld.tle.load.tle_file',return_value=sats), \
             patch('sim_mld.tle.SatrecArray',return_value=fake_array):
            _, kept, metadata = load_tle_valid_at_times('fixture.tle',dates)
        self.assertEqual([s.model.satnum for s in kept],[0])
        self.assertEqual(metadata['excluded_ids'],[1,2,3])


if __name__ == '__main__':
    unittest.main()
