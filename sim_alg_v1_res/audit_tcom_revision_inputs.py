"""Recheck archived revision comparisons after finding LCT flow-pair drift."""
import csv,json
from pathlib import Path
from collections import defaultdict
BASE=Path(__file__).resolve().parent

def rows(path):return list(csv.DictReader(path.open()))
def groups(data,key):
    out=defaultdict(list)
    for r in data:out[r[key]].append(r)
    return out

def audit():
    report={}
    root=BASE/'test_tcom_multishell/comment-2-1-20260922-ail'
    data=[json.loads(f.read_text()) for f in (root/'results').glob('*.json')]
    assert len(data)==270
    for key,rr in groups(data,'case').items():
        assert len(rr)==5
        assert len({json.dumps(r['fingerprint'],sort_keys=True) for r in rr})==1
        for r in rr:assert abs(r['served_ratio']-r['served_throughput_gbps']/r['offered_demand_gbps'])<1e-10
    report['multi_shell']={'passed':True,'evaluations':270,'matched_within_snapshot':True}
    root=BASE/'test_tcom_atp_transition/comment-1-1-20260924-ail'
    data=rows(root/'results.csv');assert len(data)==1295
    for key,rr in groups(data,'offset_seconds').items():
        assert len(rr)==35 and len({r['input_sha256'] for r in rr})==1
        for method,mm in groups(rr,'method').items():
            assert len({r['matching_sha256'] for r in mm})==1
            for r in mm:assert float(r['atp_served_throughput_gbps'])<=float(r['original_served_throughput_gbps'])+1e-6
    report['atp']={'passed':True,'replay_rows':1295,'matched_methods_and_delays':True,'same_decisions_across_delays':True}
    root=BASE/'test_tcom_flow_delay/comment-3-1-20260925-ail'
    data=rows(root/'snapshots.csv');assert len(data)==185
    for key,rr in groups(data,'offset_seconds').items():assert len(rr)==5 and len({r['input_sha256'] for r in rr})==1
    report['delay']={'passed':True,'evaluations':185,'matched_within_snapshot':True}
    root=BASE/'test_tcom_gateway_count/comment-3-4-20260926-ail'
    data=rows(root/'results.csv')
    for key,rr in groups(data,'gateway_count').items():
        assert len(rr)==5 and len({r['input_sha256'] for r in rr})==1
        for r in rr:assert abs(float(r['served_ratio'])-float(r['served_throughput_gbps'])/float(r['residual_demand_gbps']))<1e-10
    report['gateway']={'passed':True,'evaluations':len(data),'matched_within_count':True,'note':'Gateway changes legitimately alter supplying satellites and eligible flow pairs.'}
    report['scope']='Checks saved revision input fingerprints and metric consistency, not a fresh rerun of every historical figure. Original Fig. 8 changes topology and reselects gateway sources; it is not a fixed-flow-pair hardware-only sensitivity experiment. Its full original raw states are unavailable locally.'
    return report
if __name__=='__main__':print(json.dumps(audit(),indent=2))
