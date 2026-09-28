"""Diagnostic iteration sweep; independent conversion copies preserve optimizer state."""
import argparse, copy, csv, json, os, time
from pathlib import Path
import numpy as np
import torch
import test_tcom_lct_count as base
from flow_delay_metrics import route_delays

def validate(sim,r):
    s=sim.solver;selected=np.asarray(r['connected_lct'],dtype=int).reshape(-1,2)
    edges={tuple(sorted(e)) for e in sim.filtered_lct_pair_expanded}
    assert len(set(selected.ravel()))==selected.size
    assert all(tuple(sorted(e)) in edges for e in selected)
    rates=np.asarray(r['rates']);assert np.isfinite(rates).all() and np.all(rates>=-1e-8)
    sat=selected//4
    capacity=s.compute_capacity(np.linalg.norm(sim.positions[sat[:,0]]-sim.positions[sat[:,1]],axis=1))
    caps={};loads={}
    for (u,v),c in zip(sat,capacity):
        for e in ((int(u),int(v)),(int(v),int(u))):caps[e]=caps.get(e,0.)+float(c)
    for path,n,rate in zip(r['paths_all'],r['lengths'],rates):
        if rate<=0:continue
        for u,v in zip(path[:int(n)-1],path[1:int(n)]):
            e=(int(u),int(v));loads[e]=loads.get(e,0.)+float(rate)
    assert all(v<=caps.get(e,0)+1e-5 for e,v in loads.items())
    route_delays(sim.positions,r['paths_all'],r['lengths'],rates,s.data_source,s.data_target,sat)
    assert np.all(np.bincount(s.data_source,weights=rates,minlength=sim.n_sat)<=s.forward_traffic_capacity+1e-5)
    assert np.all(np.bincount(s.data_target,weights=rates,minlength=sim.n_sat)<=s.forward_traffic_demand+1e-5)
    assert abs(float(rates.sum())-r['primal_objective'])<1e-7
    return selected,capacity

def run(args):
    root=Path(args.atp_root);out=Path(args.output)/str(args.count);out.mkdir(parents=True,exist_ok=True)
    base.setup_env(root);np.random.seed(0);torch.manual_seed(0)
    sim,_=base.make_sim(root,0,args.count,'DuJo');fp=base.atp.input_fingerprint(sim)
    reference=json.loads((root/'lct-fixed-pairs/cases'/f'0_{args.count}_DuJo.json').read_text())
    assert fp==reference['input_sha256']
    base.write_json(out/'settings.json',dict(lct_count=args.count,seed=0,steps=args.steps,conversion_every=100,checkpoints=[500,1000,2000,5000],input_sha256=fp,source_sha256=base.sha(__file__),base_source_sha256=base.sha(base.__file__),common_source_sha256=base.sha(base.common.__file__),evaluation='deepcopy conversion; best only among sampled iterates',reference=reference,optimizer_alpha=sim.solver.ALPHA,optimizer_beta=sim.solver.BETA))
    rows=[];best=-1.;best_step=None;opt_seconds=0.;start=time.perf_counter()
    for k in range(1,args.steps+1):
        t=time.perf_counter();sim.solver.update_step_rates_prices();opt_seconds+=time.perf_counter()-t
        if k%100:continue
        # Conversion refreshes traffic and may alter solver scratch arrays. Isolate it.
        probe=copy.copy(sim);probe.solver=copy.deepcopy(sim.solver)
        r=base.common.collect_dujo_result(probe,0)
        assert base.atp.input_fingerprint(probe)==fp==base.atp.input_fingerprint(sim)
        selected,capacity=validate(probe,r);value=r['primal_objective']
        if k==500:np.testing.assert_allclose(value,reference['served_throughput_gbps'],atol=1e-7,rtol=0)
        if value>best:best=value;best_step=k
        row=dict(lct_count=args.count,iteration=k,throughput_gbps=value,served_ratio=value/probe.solver.forward_traffic_demand.sum(),dual_objective=r['dual_objective'],best_sampled_gbps=best,best_sampled_iteration=best_step,optimization_seconds=opt_seconds,wall_seconds=time.perf_counter()-start,input_sha256=fp)
        rows.append(row)
        np.savez_compressed(out/f'iterate_{k}.npz',rates=r['rates'],paths=r['paths_all'],lengths=r['lengths'],selected_lct=selected,selected_capacity=capacity,sources=probe.solver.data_source,targets=probe.solver.data_target)
        base.write_json(out/'trace.json',rows)
        with (out/'trace.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerows(rows)
        print(json.dumps(row),flush=True)
    base.write_json(out/'complete.json',dict(passed=True,samples=len(rows),baseline_500_reproduced=True,checks=['fixed input fingerprints','matching and route feasibility','source/target budgets','directed link capacities','finite nonnegative rates'],last=rows[-1]))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--atp-root',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,choices=[2,3,4],required=True);p.add_argument('--steps',type=int,default=5000);run(p.parse_args())
