#!/usr/bin/env python3
"""TCOM Comment 2.1: validated, resumable 3 x 3 x 6 snapshot benchmark.

Run --stage geometry, then --stage pilot, then --stage full. Outputs are keyed
by case/method; full reuses only this run's validated pilot rows, never old data.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sgp4.api import SatrecArray, jday
from skyfield.api import load
from skyfield.sgp4lib import TEME
import torch
import tcom_revision_common as common
from sim_mld.tle import load_tle_valid_at_times, select_shifted_shell_block

OFFSETS = [0, 30, 60, 90, 120, 180]
CONFIGS = {'shell53': [(53.22, 15.087, 1000)],
           'shell43': [(43.00, 15.024, 1000)],
           'mixed': [(53.22, 15.087, 500), (43.00, 15.024, 500)]}
CHECKPOINT = ('sim_alg_j1_res/train_rl_starlink_1000_pg/'
              'train_rl_starlink_1000_pg-2026-April-10-13-48-52-ail/rl_model.model_final_pg.pt')
REFERENCE = ('sim_alg_v1_res/test_tcom_shell_time_load_scaling/'
             'merged-2026-April-11-19-42-59-ail/shell_metadata.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def propagate(sat_array, t):
    jd, fraction = jday(*t.utc)
    error, p, v = sat_array.sgp4(np.array([jd]), np.array([fraction]))
    if np.any(error) or not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError('Invalid propagation at evaluation epoch')
    rotation = TEME.rotation_at(t).T
    return p[:, 0, :] @ rotation.T, v[:, 0, :] @ rotation.T


class Snapshot(common.SnapshotDualSimulation):
    """Use validated UTC SGP4 propagation without changing legacy experiments."""
    def _update_satellite_positions(self):
        p, v = propagate(self.sat_array, self.get_simulation_time())
        self.positions, self.velocities = p / self.EARTH_RADIUS, v / self.EARTH_RADIUS


def fingerprint(sim):
    arrays = {'positions': sim.positions, 'velocities': sim.velocities,
              'satellite_pairs': sim.filtered_sat_pair_repeated,
              'terminal_pairs': sim.filtered_lct_pair_expanded,
              'terminal_mask': sim.lct_mask, 'view_cos': sim.solver.possible_lct_pair_expanded_view_cos,
              'sources': sim.solver.data_source, 'targets': sim.solver.data_target,
              'supply': sim.solver.forward_traffic_capacity, 'demand': sim.solver.forward_traffic_demand,
              'gateways': sim.terrain.ground_station_positions_rotated}
    result = {}
    for name, value in arrays.items():
        a = np.ascontiguousarray(value)
        if not np.isfinite(a).all():
            raise ValueError(f'Nonfinite input {name}')
        result[name] = hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()
    return result


def make_sim(ts, sats, offset, solver):
    sim = Snapshot(ts, SatrecArray([s.model for s in sats]), common.snapshot_time_scale(ts, offset),
                   traffic_seed=0, dujo_traffic_mode='fixed', dujo_eval_mode='last')
    common.configure_simulation(sim, solver, 1e-4)
    sim.update_solver_traffic_info(seed=0)
    sim._loaded_traffic_seed = 0
    return sim


def unique_pairs(pairs):
    a = np.asarray(list(pairs) if isinstance(pairs, set) else pairs, dtype=np.int64).reshape(-1, 2)
    return {tuple(sorted(p)) for p in a.tolist()}


def cross_count(pairs, clusters):
    return sum(clusters[i] != clusters[j] for i, j in unique_pairs(pairs))


def build_cases():
    ts = load.timescale(builtin=True)
    times = [common.snapshot_time_scale(ts, t) for t in OFFSETS]
    ts, catalogue, validity = load_tle_valid_at_times(ROOT / 'starlink_16_jul_2025_1600.tle', times)
    cases, selections = [], {}
    for config, specs in CONFIGS.items():
        for selection in range(3):
            chosen, labels, parts = [], [], []
            for cluster, (inc, motion, n) in enumerate(specs):
                idx, info = select_shifted_shell_block(catalogue, inc, motion, n, selection)
                chosen.extend(catalogue[i] for i in idx)
                labels.extend([cluster] * n)
                parts.append(info)
            ids = [s.model.satnum for s in chosen]
            assert len(ids) == len(set(ids)) == 1000
            key = f'{config}-s{selection}'
            selections[key] = {'parts': parts, 'satellite_ids': ids,
                               'satellite_names': [s.name for s in chosen], 'clusters': labels}
            for offset in OFFSETS:
                cases.append((f'{key}-t{offset:03}', config, selection, offset, chosen, labels))
    reference = json.loads((ROOT / REFERENCE).read_text())['selected_satellites']
    original = selections['shell53-s0']['satellite_names']
    overlap = {}
    for config in CONFIGS:
        sets = [set(selections[f'{config}-s{s}']['satellite_ids']) for s in range(3)]
        overlap[config] = [[len(a & b) for b in sets] for a in sets]
    metadata = {'validity': validity, 'selections': selections, 'overlap_counts': overlap,
                'original_selection_comparison': {'same_order': original == reference,
                  'added_names': sorted(set(original) - set(reference)),
                  'removed_names': sorted(set(reference) - set(original))},
                'offset_minutes': OFFSETS, 't0_utc': common.T0_UTC.isoformat(),
                'steps': 500, 'traffic_seed': 0, 'traffic_mode': 'fixed', 'eval_mode': 'last',
                'active_user_fraction': 1e-4, 'methods': common.HEADLINE_METHODS,
                'checkpoint': CHECKPOINT, 'checkpoint_sha256': sha(ROOT / CHECKPOINT),
                'tle_sha256': sha(ROOT / 'starlink_16_jul_2025_1600.tle'),
                'data_sha256': {n: sha(ROOT / n) for n in ['satnogs_locations.csv', 'population_density_texture.npy']},
                'source_sha256': {str(p.relative_to(ROOT)): sha(p) for d in ['sim_mld', 'sim_src']
                                  for p in sorted((ROOT / d).rglob('*.py'))}}
    for name in ['test_tcom_multishell.py', 'tcom_revision_common.py']:
        metadata['source_sha256']['sim_alg_v1_res/' + name] = sha(ROOT / 'sim_alg_v1_res' / name)
    revision = ROOT / 'source_revision.json'
    metadata['source_revision'] = json.loads(revision.read_text()) if revision.exists() else {
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    return ts, cases, metadata


def geometry(ts, case):
    key, config, selection, offset, sats, labels = case
    t = common.snapshot_time_scale(ts, offset)
    p, v = propagate(SatrecArray([s.model for s in sats]), t)
    references = [sat.at(t) for sat in sats]
    p_err = float(np.max(np.abs(p - np.array([s.position.km for s in references]))))
    v_err = float(np.max(np.abs(v - np.array([s.velocity.km_per_s for s in references]))))
    assert p_err < 1e-6 and v_err < 1e-8, (p_err, v_err)
    sim = make_sim(ts, sats, offset, common.mr_solver())
    pairs = unique_pairs(sim.filtered_sat_pair_repeated)
    # Check the original geometry constraints through the generated terminal pairs.
    assert all(i != j and 0 <= i < 1000 and 0 <= j < 1000 for i, j in pairs)
    return {'case': key, 'config': config, 'selection': selection, 'offset_minutes': offset,
            'fingerprint': fingerprint(sim), 'candidate_pairs': len(pairs),
            'candidate_inter_shell_pairs': cross_count(pairs, labels),
            'skyfield_position_error_km': p_err, 'skyfield_velocity_error_km_s': v_err}


def evaluate(ts, case, method, expected):
    key, config, selection, offset, sats, labels = case
    torch.manual_seed(0)
    np.random.seed(0)
    start = time.perf_counter()
    solver = common.RevisionLpdSolver() if method == 'DuJo' else (
        common.DrLSnapshotSolver() if method == 'DRL' else common.mr_solver())
    sim = make_sim(ts, sats, offset, solver)
    assert fingerprint(sim) == expected['fingerprint'], f'Input mismatch {key}/{method}'
    if not len(sim.filtered_lct_pair_expanded):
        result = {'rates': np.empty(0), 'lengths': np.empty(0), 'connected_sat': np.empty((0, 2)),
                  'connected_lct': np.empty((0, 2)), 'primal_objective': 0., 'dual_objective': None}
    elif method == 'DuJo':
        # Identical fixed-traffic dual updates, without GUI/thread bookkeeping.
        for _ in range(500):
            sim.run_step()
        result = common.collect_dujo_result(sim, traffic_seed=0)
    else:
        if method == 'DRL':
            solver.load_drl(str(ROOT / CHECKPOINT), variant='pg')
            solver.infer_drl()
            values = solver.get_prim_objective(with_rates=True)
        else:
            values = solver.get_prim_objective_heuristic(with_rates=True, seed=0,
                                                       **common.HEURISTIC_METHOD_SPECS[method])
        objective, rates, routing, (costs, lengths, paths), connected, terminals = values
        result = {'primal_objective': float(-objective), 'dual_objective': None,
                  'rates': rates, 'lengths': lengths, 'connected_sat': connected, 'connected_lct': terminals}
    assert fingerprint(sim) == expected['fingerprint'], f'Inputs changed {key}/{method}'
    pairs = unique_pairs(result['connected_sat'])
    assert pairs <= unique_pairs(sim.filtered_sat_pair_repeated), 'Infeasible satellite matching'
    selected_lct = np.asarray(result['connected_lct'], dtype=int).reshape(-1, 2)
    assert unique_pairs(selected_lct) <= unique_pairs(sim.filtered_lct_pair_expanded), 'Infeasible LCT matching'
    assert len(set(selected_lct.ravel())) == selected_lct.size, 'Reused LCT'
    metrics = common.compute_metrics(sim, method, result['rates'], result['lengths'], result['connected_sat'])
    offered, served = metrics['offered_demand_gbps'], metrics['served_throughput_gbps']
    assert np.isfinite(result['rates']).all() and np.min(result['rates'], initial=0) >= -1e-6
    assert -1e-6 <= served <= offered + max(1e-5, offered * 1e-7)
    assert served <= float(sim.solver.forward_traffic_capacity.sum()) + 1e-5
    assert abs(served - result['primal_objective']) < max(1e-5, served * 1e-7)
    return {**metrics, 'case': key, 'config': config, 'selection': selection, 'offset_minutes': offset,
            'snapshot_utc': common.snapshot_datetime(offset).isoformat(),
            'evaluation_seconds': time.perf_counter() - start,
            'candidate_inter_shell_pairs': expected['candidate_inter_shell_pairs'],
            'selected_inter_shell_pairs': cross_count(pairs, labels), 'fingerprint': expected['fingerprint']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['geometry', 'pilot', 'full'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', choices=list(CONFIGS))
    parser.add_argument('--selection', type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS', '1')))
    ts, cases, metadata = build_cases()
    manifest = args.output / 'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == metadata, 'Run provenance changed; use a new output directory'
    else:
        write_json(manifest, metadata)
    if args.stage != 'geometry':
        assert all((args.output / 'geometry' / f'{c[0]}.json').exists() for c in cases), 'Geometry gate incomplete'
    if args.stage == 'full':
        assert all((args.output / 'results' / f'{c}-s0-t000-{m}.json').exists()
                   for c in CONFIGS for m in common.HEADLINE_METHODS), 'Pilot gate incomplete'
    for case in cases:
        key, config, selection, offset, _, _ = case
        if args.config and config != args.config:
            continue
        if args.selection is not None and selection != args.selection:
            continue
        if args.stage == 'pilot' and (selection != 0 or offset != 0):
            continue
        geometry_path = args.output / 'geometry' / f'{key}.json'
        if args.stage == 'geometry':
            if not geometry_path.exists():
                with (args.output / 'geometry.log').open('a') as log, contextlib.redirect_stdout(log):
                    result = geometry(ts, case)
                write_json(geometry_path, result)
            print(f'GEOMETRY {key}', flush=True)
        else:
            expected = json.loads(geometry_path.read_text())
            for method in common.HEADLINE_METHODS:
                path = args.output / 'results' / f'{key}-{method}.json'
                if path.exists():
                    continue
                log_path = args.output / f'{key}-{method}.log'
                with log_path.open('w') as log, contextlib.redirect_stdout(log):
                    result = evaluate(ts, case, method, expected)
                write_json(path, result)
                print(f'DONE {key} {method} {result["served_throughput_gbps"]:.3f} Gbps '
                      f'{result["evaluation_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
