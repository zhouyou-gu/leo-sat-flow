"""Compute independent DuJo snapshots for chronological ATP replay on one host."""
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import test_tcom_atp_transition as replay
import tcom_revision_common as common


class FinalOnlySimulation(common.SnapshotDualSimulation):
    """Keep all dual updates; omit unused intermediate objective diagnostics."""
    def run_step(self):
        if self.filtered_lct_pair_expanded.size == 0:
            return
        seed = self.N_STEP if self.dujo_traffic_mode == "legacy" else self.traffic_seed
        if self._loaded_traffic_seed != seed:
            self.update_solver_traffic_info(seed=seed)
            self._loaded_traffic_seed = seed
        self.solver.update_step_rates_prices()



def compute(offset):
    ts, _, array, _ = replay.load_recorded_population(os.environ['TCOM_REVISION_POPULATION_METADATA'])
    common.SnapshotDualSimulation = FinalOnlySimulation
    started = time.perf_counter()
    result = replay.evaluate_method_seconds('DuJo', ts, array, offset,
        active_user_percentage=0.0001, traffic_seed=0, dujo_steps=500,
        dujo_traffic_mode='legacy', dujo_eval_mode='last', include_simulation=True)
    sim = result['_simulation']
    # Retain every input used by the ATP replay, without nonserializable SGP4 handles.
    result['_simulation'] = SimpleNamespace(
        solver=sim.solver, positions=sim.positions, velocities=sim.velocities,
        filtered_lct_pair_expanded=sim.filtered_lct_pair_expanded, lct_mask=sim.lct_mask,
        terrain=SimpleNamespace(ground_station_positions_rotated=sim.terrain.ground_station_positions_rotated))
    assert replay.input_fingerprint(result['_simulation']) == replay.input_fingerprint(sim)
    out = Path(os.environ['TCOM_REVISION_SNAPSHOT_CACHE']) / f'DuJo_{offset}.pkl'
    with out.with_suffix('.partial').open('wb') as fp:
        pickle.dump((time.perf_counter()-started, result), fp)
    out.with_suffix('.partial').replace(out)
    return offset


if __name__ == '__main__':
    Path(os.environ['TCOM_REVISION_SNAPSHOT_CACHE']).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=int(os.getenv('TCOM_REVISION_WORKERS', '8'))) as pool:
        for offset in pool.map(compute, replay.default_offsets()):
            print('completed', offset, flush=True)
