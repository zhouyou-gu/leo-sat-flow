"""Audit every sampled feasible iterate and summarize the requested budgets."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from flow_delay_metrics import route_delays

def audit(root,reference):
    summary=[]
    for count in [2,3,4]:
        folder=root/str(count);done=json.loads((folder/'complete.json').read_text());assert done['passed']
        trace=json.loads((folder/'trace.json').read_text());assert [r['iteration'] for r in trace]==list(range(100,5001,100))
        refrow=json.loads((reference/'cases'/f'0_{count}_DuJo.json').read_text())
        with np.load(reference/'cases'/f'0_{count}_DuJo.npz') as b:
            edges={tuple(sorted(e)) for e in b['candidate_lct']};best=-1.
            for row in trace:
                assert row['input_sha256']==refrow['input_sha256']
                with np.load(folder/f"iterate_{row['iteration']}.npz") as a:
                    for k in ['sources','targets']:np.testing.assert_array_equal(a[k],b[k])
                    selected=a['selected_lct'];sat=selected//4
                    assert len(set(selected.ravel()))==selected.size
                    assert all(tuple(sorted(e)) in edges for e in selected)
                    rates=a['rates'];assert np.isfinite(rates).all() and np.all(rates>=-1e-8)
                    route_delays(b['positions'],a['paths'],a['lengths'],rates,a['sources'],a['targets'],sat)
                    caps={};loads={}
                    for (u,v),c in zip(sat,a['selected_capacity']):
                        for e in ((int(u),int(v)),(int(v),int(u))):caps[e]=caps.get(e,0.)+float(c)
                    for path,n,rate in zip(a['paths'],a['lengths'],rates):
                        if rate<=0:continue
                        for u,v in zip(path[:int(n)-1],path[1:int(n)]):
                            e=(int(u),int(v));loads[e]=loads.get(e,0.)+float(rate)
                    assert all(v<=caps.get(e,0.)+1e-5 for e,v in loads.items())
                    for k,budget in [('sources','supply'),('targets','demand')]:
                        assert np.all(np.bincount(a[k],weights=rates,minlength=len(b['positions']))<=b[budget]+1e-5)
                    np.testing.assert_allclose(rates.sum(),row['throughput_gbps'],atol=1e-8,rtol=0)
                    np.testing.assert_allclose(rates.sum()/b['demand'].sum(),row['served_ratio'],atol=1e-10,rtol=0)
                    if row['iteration']==500:
                        for k in ['rates','paths','lengths','selected_lct']:np.testing.assert_array_equal(a[k],b[k])
                    best=max(best,float(rates.sum()));assert abs(best-row['best_sampled_gbps'])<1e-8
                    if row['iteration'] in [500,1000,2000,5000]:summary.append(row)
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    (root/'independent_audit.json').write_text(json.dumps(dict(passed=True,sampled_iterates=150,baseline500='rates, routes and matching exactly reproduced for all counts',best='among every-100-step samples only'),indent=2))
    for r in summary:print(r['lct_count'],r['iteration'],round(r['throughput_gbps'],3),round(r['best_sampled_gbps'],3),r['best_sampled_iteration'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('reference',type=Path);a=p.parse_args();audit(a.root,a.reference)
