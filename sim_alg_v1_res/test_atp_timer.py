"""Focused checks for the ATP replay state transitions (no optimization run)."""
import unittest
from test_tcom_atp_transition import update_atp_state


class TimerTests(unittest.TestCase):
    def test_preacquired_and_multisnapshot_completion(self):
        old, new = (0, 1), (2, 3)
        usable, _, state = update_atp_state(0, {old}, set(), {}, True, 90)
        self.assertEqual(usable, {old})
        previous = {old}
        for t in range(10, 110, 10):
            usable, _, state = update_atp_state(t, {old, new}, previous, state, False, 90)
            self.assertEqual(usable, {old, new} if t == 100 else {old})
            previous = {old, new}

    def test_drop_and_reselection_restarts(self):
        link = (0, 1)
        _, _, state = update_atp_state(10, {link}, set(), {}, False, 50)
        _, _, state = update_atp_state(40, set(), {link}, state, False, 50)
        self.assertNotIn(link, state)
        usable, _, state = update_atp_state(50, {link}, set(), state, False, 50)
        self.assertEqual(usable, set())
        usable, _, state = update_atp_state(90, {link}, {link}, state, False, 50)
        self.assertEqual(usable, set())
        usable, _, _ = update_atp_state(100, {link}, {link}, state, False, 50)
        self.assertEqual(usable, {link})

    def test_zero_delay_and_empty_topology(self):
        link = (0, 1)
        usable, _, state = update_atp_state(10, {link}, set(), {}, False, 0)
        self.assertEqual(usable, {link})
        usable, _, state = update_atp_state(20, set(), {link}, state, False, 0)
        self.assertEqual((usable, state), (set(), {}))


if __name__ == '__main__':
    unittest.main()
