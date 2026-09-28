"""Independently check archived installed-LCT comparison inputs and outputs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from flow_delay_metrics import route_delays

def audit(root):
    rows=list(csv.DictReader((root/'results.csv').open()))
    assert len(rows)==15
    baseline=None;previous=set();previous_selected=set()
    for count in [2,3,4]:
        subset=[r for r in rows if int(r['lct_count'])==count]
        assert len(subset)==5 and len({r['method'] for r in subset})==5
        assert len({r['input_sha256'] for r in subset})==1
        within=None
        for r in subset:
            with np.load(root/'cases'/f"0_{count}_{r['method']}.npz") as a:
                shared=[a[k] for k in ['positions','gateways','raw_user_demand','supply','demand','sources','targets']]
                if baseline is None:baseline=[x.copy() for x in shared]
                for x,y in zip(shared,baseline):np.testing.assert_allclose(x,y,atol=1e-12,rtol=0)
                inputs=[a[k] for k in ['candidate_lct','lct_mask','lct_directions','sources','targets']]
                if within is None:within=[x.copy() for x in inputs]
                for x,y in zip(inputs,within):np.testing.assert_array_equal(x,y)
                assert np.all(a['lct_mask'].sum(axis=1)==count)
                edges={tuple(sorted(e)) for e in a['candidate_lct']}
                selected=a['selected_lct'];assert len(set(selected.ravel()))==selected.size
                assert all(tuple(sorted(e)) in edges for e in selected)
                assert np.all(selected%4<count)
                sat=selected//4
                capacities={}
                for (u,v),capacity in zip(sat,a['selected_capacity']):
                    for edge in [(int(u),int(v)),(int(v),int(u))]:capacities[edge]=capacities.get(edge,0.)+float(capacity)
                loads={}
                for path,n,rate in zip(a['paths'],a['lengths'],a['rates']):
                    for u,v in zip(path[:int(n)-1],path[1:int(n)]):
                        edge=(int(u),int(v));loads[edge]=loads.get(edge,0.)+float(rate)
                assert all(value<=capacities[edge]+1e-5 for edge,value in loads.items())
                route_delays(a['positions'],a['paths'],a['lengths'],a['rates'],a['sources'],a['targets'],sat)
                served=float(a['rates'].sum());demand=float(a['demand'].sum())
                assert abs(served-float(r['served_throughput_gbps']))<1e-8
                assert abs(served/demand-float(r['served_ratio']))<1e-10
                out=np.bincount(a['sources'],weights=a['rates'],minlength=len(a['positions']))
                into=np.bincount(a['targets'],weights=a['rates'],minlength=len(a['positions']))
                assert np.all(out<=a['supply']+1e-5) and np.all(into<=a['demand']+1e-5)
        assert previous<=edges and previous_selected<=edges;previous=edges
        with np.load(root/'cases'/f"0_{count}_DuJo.npz") as a:
            previous_selected={tuple(sorted(e)) for e in a['selected_lct']}
        print(count,'LCTs PASS')
    (root/'independent_audit.json').write_text(json.dumps({'passed':True,'evaluations':15,'checks':['matched geometry/traffic/gateways','matched within-count inputs','nested candidate sets','installed masks','feasible matching/routes','throughput/ratio','source and target capacity bounds','directed link capacity bounds','previous DuJo matching remains feasible with added terminals','fixed source-target pairs across counts']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);audit(p.parse_args().results)
