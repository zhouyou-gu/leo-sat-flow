import unittest
from types import SimpleNamespace
import numpy as np
from test_tcom_served_demand_audit import flow_bound

class BoundsTests(unittest.TestCase):
    def test_capacity_cut_and_candidate_relaxation(self):
        s=SimpleNamespace(N_LCT_PER_SAT=2,positions=np.array([[0.,0,0],[1.,0,0],[2.,0,0]]),forward_traffic_capacity=np.array([10.,0,0]),forward_traffic_demand=np.array([0.,0,10.]),compute_capacity=lambda d:np.full(len(d),4.))
        selected=np.array([[0,2],[3,4]])
        self.assertAlmostEqual(flow_bound(s,selected),4.)
        self.assertAlmostEqual(flow_bound(s,np.vstack([selected,selected[:,::-1]])),4.)
        self.assertAlmostEqual(flow_bound(s,np.vstack([selected,[[1,5]]])),8.)
    def test_disconnected(self):
        s=SimpleNamespace(N_LCT_PER_SAT=2,positions=np.zeros((3,3)),forward_traffic_capacity=np.array([10.,0,0]),forward_traffic_demand=np.array([0.,0,10.]),compute_capacity=lambda d:np.full(len(d),4.))
        self.assertAlmostEqual(flow_bound(s,np.array([[0,2]])),0.)

if __name__=='__main__':unittest.main()
