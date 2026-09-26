"""Audit residual-demand accounting and relaxed capacity bounds from trusted ATP caches."""
import argparse,csv,hashlib,json,pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from sim_mld.terrain import terrain


def flow_bound(solver, terminals):
    graph=nx.DiGraph()
    terminals=np.unique(np.sort(np.asarray(terminals,dtype=int).reshape(-1,2),axis=1),axis=0)
    sat=terminals//solver.N_LCT_PER_SAT
    capacity=solver.compute_capacity(np.linalg.norm(solver.positions[sat[:,0]]-solver.positions[sat[:,1]],axis=1))
    for (i,j),cap in zip(sat,capacity):
        for u,v in [(int(i),int(j)),(int(j),int(i))]:
            old=graph[u][v]['capacity'] if graph.has_edge(u,v) else 0.
            graph.add_edge(u,v,capacity=old+float(cap))
    for i,q in enumerate(solver.forward_traffic_capacity):
        if q>0:graph.add_edge('source',i,capacity=float(q))
    for i,d in enumerate(solver.forward_traffic_demand):
        if d>0:graph.add_edge(i,'sink',capacity=float(d))
    return nx.maximum_flow_value(graph,'source','sink',flow_func=nx.algorithms.flow.preflow_push)


def main(args):
    root=Path(args.atp_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    for offset in range(0,361,10):
        cache=root/'cache'/f'DuJo_{offset}.pkl'
        with cache.open('rb') as f:_,r=pickle.load(f)
        sim=r['_simulation'];s=sim.solver;d=s.forward_traffic_demand;q=s.forward_traffic_capacity
        gw=cKDTree(sim.terrain.ground_station_positions_rotated)
        _,idx=gw.query(sim.positions,distance_upper_bound=terrain.GW_RANGE/terrain.EARTH_RADIUS)
        gateway=idx!=gw.n
        local=float(gateway.sum()*terrain.TARGET_UL_RATE-q.sum())
        demand=float(d.sum());supply=float(q.sum());served=float(r['rates'].sum())
        assert local>=0 and np.all(q[~gateway]==0)
        reachable=float(d[np.unique(s.data_target[r['lengths']>0])].sum())
        selected_bound=flow_bound(s,r['connected_lct'])
        all_bound=flow_bound(s,sim.filtered_lct_pair_expanded)
        # Capacity/route relaxation admits any supplying satellite, splitting, and all paths;
        # candidate bound additionally drops terminal-matching constraints.
        assert served<=min(reachable,selected_bound,all_bound,supply,demand)+1e-5
        assert selected_bound<=all_bound+1e-5
        row=dict(offset_seconds=offset,residual_demand_gbps=demand,residual_supply_gbps=supply,local_served_gbps=local,total_user_demand_gbps=local+demand,routed_gbps=served,residual_served_ratio=served/demand,total_served_ratio=(local+served)/(local+demand),selected_reachable_demand_gbps=reachable,selected_relaxed_flow_bound_gbps=selected_bound,candidate_relaxed_flow_bound_gbps=all_bound,cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
        rows.append(row);print(offset,json.dumps(row),flush=True)
    with (out/'snapshots.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    keys=[k for k in rows[0] if k not in ['offset_seconds','cache_sha256']]
    summary={k:{'mean':float(np.mean([r[k] for r in rows])),'min':float(min(r[k] for r in rows)),'max':float(max(r[k] for r in rows))} for k in keys}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'validation.json').write_text(json.dumps({'passed':True,'snapshots':37,'gateway_range_km':terrain.GW_RANGE,'gateway_capacity_gbps':terrain.TARGET_UL_RATE,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bounds':'Single-commodity max-flow relaxations permit any supplying satellite and splittable routing. Candidate bound additionally permits every candidate terminal pair simultaneously. Bounds do not certify optimality.'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--atp-root',required=True);p.add_argument('--output',required=True);main(p.parse_args())
