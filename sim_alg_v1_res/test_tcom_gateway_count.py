"""Gateway-count sensitivity with nested placements and fixed snapshot traffic."""
import argparse,csv,hashlib,json,os,sys,time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
import tcom_revision_common as common
import test_tcom_atp_transition as atp
from sim_mld.constellation import lat_lon_to_xyz

COUNTS=[50,100,150,200,250]
METHODS=['DuJo','DRL','SaTE','MRate','+Grid']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def nested_indices(size,seed):
    if size<400:raise ValueError('Gateway catalogue must contain at least 400 records')
    rng=np.random.default_rng(seed);original=rng.choice(np.arange(size),size=100,replace=False)
    extra=rng.choice(np.setdiff1d(np.arange(size),original),size=100,replace=False)
    first200=np.concatenate([original,extra])
    extension=rng.choice(np.setdiff1d(np.arange(size),first200),size=200,replace=False)
    return np.concatenate([first200,extension])
def setup_env(root):
    execution=json.loads((Path(root)/'results/execution.json').read_text());os.environ.update(execution['environment']);torch.set_num_threads(1)
    assert sha(execution['environment']['TCOM_REVISION_DRL_MODEL_PATH'])==execution['checkpoint_sha256']
    return execution

def make_sim(root,seed,count,method):
    ts,_,arr,_=atp.load_recorded_population(str(Path(root)/'original_metadata.json'))
    solver=common.RevisionLpdSolver() if method=='DuJo' else common.DrLSnapshotSolver() if method=='DRL' else common.mr_solver()
    sim=common.SnapshotDualSimulation(ts,arr,common.snapshot_time_scale_from_offset(ts,0),traffic_seed=0,dujo_traffic_mode='fixed',dujo_eval_mode='last')
    catalogue=pd.read_csv('satnogs_locations.csv');idx=nested_indices(len(catalogue),seed)[:count]
    sim.terrain.ground_station_positions=lat_lon_to_xyz(np.deg2rad(catalogue.iloc[idx][['lat','lng']].to_numpy()))
    common.configure_simulation(sim,solver,0.0001);sim.update_solver_traffic_info(seed=0)
    return sim,idx

def state_info(sim):
    s=sim.solver;t=sim.terrain
    _,idx=t.ground_station_positions_rotated_kd_tree.query(sim.positions,distance_upper_bound=t.GW_RANGE/t.EARTH_RADIUS)
    covered=idx!=t.ground_station_positions_rotated_kd_tree.n
    local=covered.astype(float)*t.TARGET_UL_RATE-s.forward_traffic_capacity
    raw=s.forward_traffic_demand+local
    assert np.min(local)>=-1e-8
    return dict(gateway_connected_satellites=int(covered.sum()),residual_supply_gbps=float(s.forward_traffic_capacity.sum()),residual_demand_gbps=float(s.forward_traffic_demand.sum()),local_served_gbps=float(local.sum()),total_user_demand_gbps=float(raw.sum())),raw

def prepare(args):
    root=Path(args.atp_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);execution=setup_env(root)
    base=None;geom=None;cases=[]
    for seed in args.seeds:
        prior=set();last_covered=-1
        for count in COUNTS:
            sim,idx=make_sim(root,seed,count,'MRate');info,raw=state_info(sim)
            if base is None:base=raw;geom=(sim.positions.copy(),sim.filtered_lct_pair_expanded.copy())
            np.testing.assert_allclose(raw,base,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(sim.positions,geom[0]);np.testing.assert_array_equal(sim.filtered_lct_pair_expanded,geom[1])
            assert len(set(idx))==count and prior<=set(idx);prior=set(idx)
            assert info['gateway_connected_satellites']>=last_covered;last_covered=info['gateway_connected_satellites']
            case=dict(seed=seed,gateway_count=count,gateway_indices=idx.tolist(),input_sha256=atp.input_fingerprint(sim),**info)
            write_json(out/'inputs'/f'{seed}_{count}.json',case);cases.append(case)
            if seed==0 and count==100:
                import pickle
                with (root/'cache/DuJo_0.pkl').open('rb') as f:_,cached=pickle.load(f)
                assert atp.input_fingerprint(cached['_simulation'])==case['input_sha256']
    # Verify direct fixed-traffic updates against the original diagnostic-rich loop.
    sims=[make_sim(root,args.seeds[0],100,'DuJo')[0] for _ in range(2)]
    for _ in range(5):sims[0].run_step();sims[1].solver.update_step_rates_prices()
    results=[common.collect_dujo_result(s,0) for s in sims]
    for k in ['rates','connected_lct','paths_all']:np.testing.assert_array_equal(results[0][k],results[1][k])
    assert results[0]['primal_objective']==results[1]['primal_objective']
    write_json(out/'manifest.json',dict(seeds=args.seeds,gateway_counts=COUNTS,methods=METHODS,population_metadata=str(root/'original_metadata.json'),population_sha256=sha(root/'original_metadata.json'),gateway_catalogue_sha256=sha('satnogs_locations.csv'),checkpoint_sha256=execution['checkpoint_sha256'],code_revision=args.code_revision,source_sha256=sha(__file__),dujo_steps=500,dujo_traffic_mode='fixed',dujo_eval_mode='last',traffic_seed=0,active_user_percentage=.0001,reference_time=common.T0_UTC.isoformat(),preflight='passed',direct_update_equivalence='passed',gateway_capacity_per_satellite_gbps=20,geometry_and_total_user_demand_fixed=True))
    print('PREFLIGHT PASSED',len(cases),'gateway cases',flush=True)

def evaluate(task):
    root,out,seed,count,method=task;out=Path(out);dest=out/'cases'/f'{seed}_{count}_{method}.json'
    if dest.exists():return json.loads(dest.read_text())
    setup_env(root);np.random.seed(0);torch.manual_seed(0)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
    sim,idx=make_sim(root,seed,count,method);fp=atp.input_fingerprint(sim)
    expected=json.loads((out/'inputs'/f'{seed}_{count}.json').read_text());assert fp==expected['input_sha256']
    start=time.perf_counter()
    if method=='DuJo':
        for k in range(500):
            sim.solver.update_step_rates_prices()
            if (k+1)%100==0:print('DuJo',seed,count,k+1,flush=True)
        r=common.collect_dujo_result(sim,0)
    else:
        s=sim.solver
        if method=='DRL':
            s.load_drl(os.environ['TCOM_REVISION_DRL_MODEL_PATH'],variant='pg');s.infer_drl();values=s.get_prim_objective(with_rates=True)
        else:values=s.get_prim_objective_heuristic(with_rates=True,seed=0,**common.HEURISTIC_METHOD_SPECS[method])
        obj,rates,routing,(costs,lengths,paths),sat,lct=values
        r=dict(rates=rates,paths_all=paths,lengths=lengths,connected_sat=sat,connected_lct=lct,primal_objective=float(-obj))
    runtime=time.perf_counter()-start
    assert atp.input_fingerprint(sim)==fp
    selected=np.asarray(r['connected_lct'],dtype=int).reshape(-1,2)
    possible={tuple(sorted(e)) for e in sim.filtered_lct_pair_expanded}
    assert all(tuple(sorted(e)) in possible for e in selected)
    assert len(set(selected.ravel()))==selected.size
    rates=np.asarray(r['rates']);assert np.isfinite(rates).all() and np.min(rates,initial=0)>=-1e-8
    info,raw=state_info(sim);served=float(rates.sum());assert served<=info['residual_demand_gbps']+1e-6
    assert abs(served-r['primal_objective'])<1e-7
    from flow_delay_metrics import route_delays
    route_delays(sim.positions,r['paths_all'],r['lengths'],rates,sim.solver.data_source,sim.solver.data_target,r['connected_sat'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest.with_suffix('.npz'),positions=sim.positions,gateways=sim.terrain.ground_station_positions_rotated,gateway_indices=idx,raw_user_demand=raw,supply=sim.solver.forward_traffic_capacity,demand=sim.solver.forward_traffic_demand,rates=rates,paths=r['paths_all'],lengths=r['lengths'],sources=sim.solver.data_source,targets=sim.solver.data_target,selected_lct=selected)
    row=dict(seed=seed,gateway_count=count,method=method,input_sha256=fp,served_throughput_gbps=served,served_ratio=served/info['residual_demand_gbps'] if info['residual_demand_gbps'] else 1.,evaluation_seconds=runtime,**info)
    write_json(dest,row);print('COMPLETED',seed,count,method,served,flush=True);return row

def run(args):
    out=Path(args.output);manifest=json.loads((out/'manifest.json').read_text());assert manifest['preflight']=='passed' and manifest['seeds']==args.seeds
    tasks=[(args.atp_root,args.output,seed,n,m) for seed in args.seeds for n in COUNTS for m in METHODS]
    # Schedule the slow DuJo cases first; other methods fill available slots.
    tasks.sort(key=lambda t:t[-1]!='DuJo')
    rows=[]
    manifest["execution_source_sha256"]=sha(__file__);write_json(out/"manifest.json",manifest)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as pool:
        for f in as_completed([pool.submit(evaluate,t) for t in tasks]):rows.append(f.result())
    assert len(rows)==len(args.seeds)*len(COUNTS)*len(METHODS)
    for seed in args.seeds:
        for count in COUNTS:
            subset=[r for r in rows if r['seed']==seed and r['gateway_count']==count]
            assert len(subset)==5 and len({r['input_sha256'] for r in subset})==1
    rows.sort(key=lambda r:(r['seed'],r['gateway_count'],METHODS.index(r['method'])))
    with (out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(out/'validation.json',dict(passed=True,evaluations=len(rows),matched_inputs=True,feasible_routes_and_terminals=True,throughput_bounds=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--atp-root',required=True);p.add_argument('--output',required=True);p.add_argument('--code-revision',default='20c4ace');p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--workers',type=int,default=5);p.add_argument('--prepare',action='store_true');a=p.parse_args();prepare(a) if a.prepare else run(a)
