"""Comment 3.1: reuse trusted DuJo caches and measure snapshot route delays."""
import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path
import numpy as np
import torch
import tcom_revision_common as common
import test_tcom_atp_transition as atp
from flow_delay_metrics import route_delays, weighted_stats, RATE_EPS

METHODS = ['DuJo', 'DRL', 'SaTE', 'MRate', '+Grid']
OFFSETS = list(range(0, 361, 10))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(args):
    out = Path(args.output);out.mkdir(parents=True, exist_ok=True)
    old = json.loads(Path(args.atp_root, 'results/execution.json').read_text())
    os.environ.update(old['environment'])
    torch.set_num_threads(1)
    ts, _, array, metadata = atp.load_recorded_population(os.environ['TCOM_REVISION_POPULATION_METADATA'])
    assert sha(old['environment']['TCOM_REVISION_DRL_MODEL_PATH']) == old['checkpoint_sha256']
    (out/'population.json').write_text(json.dumps(metadata, indent=2))
    baseline = {(r['method'], int(r['offset_seconds'])):r for r in csv.DictReader(open(Path(args.atp_root, 'results/results.csv'))) if int(r['atp_time_seconds']) == 0}
    rows = []
    for offset in OFFSETS:
        cache = Path(args.atp_root, 'cache', f'DuJo_{offset}.pkl')
        with cache.open('rb') as f: _, cached = pickle.load(f)
        expected = atp.input_fingerprint(cached['_simulation'])
        for method in METHODS:
            dest = out/method;dest.mkdir(exist_ok=True)
            saved = dest/f'{offset}.npz';audit = dest/f'{offset}.json'
            if saved.exists() and audit.exists():
                row = json.loads(audit.read_text());assert row['input_sha256'] == expected
                rows.append(row);continue
            np.random.seed(0);torch.manual_seed(0)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
            result = cached if method == 'DuJo' else common.evaluate_method_seconds(method, ts, array, offset, active_user_percentage=0.0001, traffic_seed=0, include_simulation=True)
            sim = result['_simulation'];assert atp.input_fingerprint(sim) == expected
            rates = np.asarray(result['rates']);lengths = np.asarray(result['lengths'])
            sources, targets = sim.solver.data_source, sim.solver.data_target
            delays = route_delays(sim.positions, result['paths_all'], lengths, rates, sources, targets, result['connected_sat'])
            total = float(rates.sum());offered = float(sim.solver.forward_traffic_demand.sum())
            assert np.isfinite(total) and -1e-7 <= total <= offered + 1e-5
            assert abs(total-result['served_throughput_gbps']) < 1e-7
            if method != 'DRL': assert abs(total-float(baseline[method,offset]['original_served_throughput_gbps'])) < 1e-7
            np.savez_compressed(saved, positions=sim.positions, paths=result['paths_all'], lengths=lengths, rates=rates, sources=sources, targets=targets, connected=result['connected_sat'], selected_lct=result['connected_lct'], delays_ms=delays)
            row = dict(method=method, offset_seconds=offset, input_sha256=expected, throughput_gbps=total, served_ratio=total/offered if offered else 1., served_flows=int((rates>RATE_EPS).sum()), reachable_flows=int((lengths>0).sum()), flow_count=len(rates), cache_sha256=sha(cache) if method=='DuJo' else '', artifact_sha256=sha(saved))
            row.update(weighted_stats(delays,rates));audit.write_text(json.dumps(row,indent=2));rows.append(row)
            print('COMPLETED',method,offset,flush=True)
    assert len(rows)==185
    summaries=[]
    for method in METHODS:
        records=[r for r in rows if r['method']==method]
        assert sorted(r['offset_seconds'] for r in records)==OFFSETS
        delays=[];weights=[]
        for offset in OFFSETS:
            with np.load(out/method/f'{offset}.npz') as a:
                checked=route_delays(a['positions'],a['paths'],a['lengths'],a['rates'],a['sources'],a['targets'],a['connected'])
                np.testing.assert_allclose(checked,a['delays_ms'],equal_nan=True)
                delays.append(checked);weights.append(a['rates'])
        summary=dict(method=method, snapshots=37, **weighted_stats(np.concatenate(delays),np.concatenate(weights)), throughput_gbps=float(np.mean([r['throughput_gbps'] for r in records])), served_ratio=float(np.mean([r['served_ratio'] for r in records])))
        summaries.append(summary)
    csv_write(out/'snapshots.csv',rows);csv_write(out/'summary.csv',summaries)
    manifest=dict(validation='passed',snapshot_evaluations=len(rows),offsets_seconds=OFFSETS,seed=0,active_user_percentage=0.0001,dujo_protocol='cached 500 legacy/last, evaluation seed 0',dujo_no_reoptimization=True,checkpoint_sha256=old['checkpoint_sha256'],tle_sha256=metadata['tle_sha256'],earth_radius_km=6371.,speed_of_light_km_s=299792.458,rate_epsilon_gbps=RATE_EPS,initial_snapshot_included=True,atp_overhead=False,source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*flow_delay*.py')},base_code_revision=args.code_revision,torch_cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    (out/'validation.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--atp-root',required=True);p.add_argument('--output',required=True);p.add_argument('--code-revision',required=True);run(p.parse_args())
