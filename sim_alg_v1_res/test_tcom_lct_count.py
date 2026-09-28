"""Installed-terminal sensitivity with fixed snapshot geometry and traffic."""
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

COUNTS=[2,3,4]
METHODS=['DuJo','DRL','SaTE','MRate','+Grid']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
class InstalledLctSimulation(common.SnapshotDualSimulation):
    def __init__(self,*args,installed_lcts,for_half_angle=60.,**kwargs):
        self.installed_lcts=installed_lcts
        self.FOR_THETA_HALF=float(for_half_angle)
        super().__init__(*args,**kwargs)
    def update_solver_traffic_info(self,seed=0):
        super().update_solver_traffic_info(seed=seed)
        if hasattr(self,'fixed_sources'):
            self.solver.data_source=self.fixed_sources.copy()
            self.solver.data_target=self.fixed_targets.copy()
            self.solver.s_t_traffic_rates=np.zeros(len(self.fixed_sources),dtype=np.float32)
    def config_l_mask(self,seed=0):
        self.lct_mask=np.zeros((self.n_sat,self.N_LCT_PER_SAT),dtype=np.float32)
        self.lct_mask[:,:self.installed_lcts]=1.

def setup_env(root):
    execution=json.loads((Path(root)/'results/execution.json').read_text());os.environ.update(execution['environment']);torch.set_num_threads(1)
    assert sha(execution['environment']['TCOM_REVISION_DRL_MODEL_PATH'])==execution['checkpoint_sha256']
    return execution

def make_sim(root,seed,count,method,for_half_angle=60.):
    ts,_,arr,_=atp.load_recorded_population(str(Path(root)/'original_metadata.json'))
    solver=common.RevisionLpdSolver() if method=='DuJo' else common.DrLSnapshotSolver() if method=='DRL' else common.mr_solver()
    sim=InstalledLctSimulation(ts,arr,common.snapshot_time_scale_from_offset(ts,0),installed_lcts=count,for_half_angle=for_half_angle,traffic_seed=0,dujo_traffic_mode='fixed',dujo_eval_mode='last')
    common.configure_simulation(sim,solver,0.0001);sim.update_solver_traffic_info(seed=0)
    # Freeze gateway-source assignments to the two-LCT reference for every layout.
    if count!=2 or for_half_angle!=60.:
        reference,_=make_sim(root,seed,2,'MRate')
        sim.fixed_sources=reference.solver.data_source.copy()
        sim.fixed_targets=reference.solver.data_target.copy()
        sim.solver.data_source=sim.fixed_sources.copy()
        sim.solver.data_target=sim.fixed_targets.copy()
        sim.solver.s_t_traffic_rates=np.zeros_like(reference.solver.s_t_traffic_rates)
    return sim,np.arange(count)

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
    base=None;previous=set()
    for count in COUNTS:
        sim,_=make_sim(root,0,count,'MRate',args.for_half_angle);info,raw=state_info(sim)
        before=atp.input_fingerprint(sim);sim.update_solver_traffic_info(seed=0);assert atp.input_fingerprint(sim)==before
        assert np.all(sim.lct_mask.sum(axis=1)==count)
        np.testing.assert_allclose(np.linalg.norm(sim.lct_directions,axis=2),1,atol=1e-6)
        np.testing.assert_allclose(sim.lct_directions[:,0],-sim.lct_directions[:,1],atol=1e-6)
        np.testing.assert_allclose(sim.lct_directions[:,2],-sim.lct_directions[:,3],atol=1e-6)
        np.testing.assert_allclose((sim.lct_directions[:,0]*sim.lct_directions[:,2]).sum(axis=1),0,atol=1e-6)
        shared=[sim.positions,sim.velocities,raw,sim.solver.forward_traffic_capacity,sim.solver.forward_traffic_demand,sim.terrain.ground_station_positions_rotated,sim.solver.data_source,sim.solver.data_target]
        if base is None:base=[x.copy() for x in shared]
        for x,y in zip(shared,base):np.testing.assert_allclose(x,y,atol=1e-12,rtol=0)
        edges={tuple(sorted(e)) for e in sim.filtered_lct_pair_expanded}
        assert previous<=edges;previous=edges
        assert all(a%4<count and b%4<count for a,b in edges)
        if count==2 and args.for_half_angle==60.:
            old=json.loads((root/'gateway-results/inputs/0_100.json').read_text())
            assert old['input_sha256']==atp.input_fingerprint(sim)
        write_json(out/'inputs'/f'0_{count}.json',dict(lct_count=count,input_sha256=atp.input_fingerprint(sim),candidate_links=len(edges),**info))
    write_json(out/'manifest.json',dict(seeds=args.seeds,lct_counts=COUNTS,methods=METHODS,population_sha256=sha(root/'original_metadata.json'),checkpoint_sha256=execution['checkpoint_sha256'],source_sha256=sha(__file__),code_revision=args.code_revision,dujo_steps=500,dujo_traffic_mode='fixed',dujo_eval_mode='last',traffic_seed=0,active_user_percentage=.0001,reference_time=common.T0_UTC.isoformat(),preflight='passed',mounting_order=['front','back','right','left'],gateway_count=100,fixed_geometry_traffic_gateways=True,nested_candidate_links=True,flow_pair_policy='fixed_two_lct_reference',for_half_angle_deg=args.for_half_angle,for_full_angle_deg=2*args.for_half_angle))
    print('PREFLIGHT PASSED',flush=True)

def evaluate(task):
    root,out,seed,count,method=task;out=Path(out);dest=out/'cases'/f'{seed}_{count}_{method}.json'
    if dest.exists():
        row=json.loads(dest.read_text());expected=json.loads((out/'inputs'/f'{seed}_{count}.json').read_text())
        assert row['input_sha256']==expected['input_sha256'], 'Stale cached inputs'
        return row
    setup_env(root);np.random.seed(0);torch.manual_seed(0)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
    half=json.loads((out/'manifest.json').read_text()).get('for_half_angle_deg',60.)
    sim,idx=make_sim(root,seed,count,method,half);fp=atp.input_fingerprint(sim)
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
    np.savez_compressed(dest.with_suffix('.npz'),positions=sim.positions,gateways=sim.terrain.ground_station_positions_rotated,lct_mask=sim.lct_mask,lct_directions=sim.lct_directions,candidate_lct=sim.filtered_lct_pair_expanded,raw_user_demand=raw,supply=sim.solver.forward_traffic_capacity,demand=sim.solver.forward_traffic_demand,rates=rates,paths=r['paths_all'],lengths=r['lengths'],sources=sim.solver.data_source,targets=sim.solver.data_target,selected_lct=selected,selected_capacity=sim.solver.compute_capacity(np.linalg.norm(sim.positions[selected[:,0]//4]-sim.positions[selected[:,1]//4],axis=1)))
    row=dict(seed=seed,lct_count=count,method=method,input_sha256=fp,served_throughput_gbps=served,served_ratio=served/info['residual_demand_gbps'] if info['residual_demand_gbps'] else 1.,evaluation_seconds=runtime,candidate_links=len(possible),selected_links=len(selected),**info)
    write_json(dest,row);print('COMPLETED',seed,count,method,served,flush=True);return row

def run(args):
    out=Path(args.output);manifest=json.loads((out/'manifest.json').read_text());assert manifest['preflight']=='passed' and manifest['seeds']==args.seeds and manifest['flow_pair_policy']=='fixed_two_lct_reference'
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
            subset=[r for r in rows if r['seed']==seed and r['lct_count']==count]
            assert len(subset)==5 and len({r['input_sha256'] for r in subset})==1
    rows.sort(key=lambda r:(r['seed'],r['lct_count'],METHODS.index(r['method'])))
    with (out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(out/'validation.json',dict(passed=True,evaluations=len(rows),matched_inputs=True,feasible_routes_and_terminals=True,throughput_bounds=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--atp-root',required=True);p.add_argument('--output',required=True);p.add_argument('--code-revision',default='6609a38');p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--workers',type=int,default=5);p.add_argument('--prepare',action='store_true');p.add_argument('--for-half-angle',type=float,default=60.);a=p.parse_args();prepare(a) if a.prepare else run(a)
