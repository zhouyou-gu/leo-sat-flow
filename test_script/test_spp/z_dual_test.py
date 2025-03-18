import random
import time
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from test_script.test_spp.spp_graph import DirectedWeightedGraphNX
from test_script.test_spp.z_spp_solver import ShortestPathSolver

from sim_src.util import STATS_OBJECT

# -------------------------------------------------------------------
# Testing and Performance Measurement Script
# -------------------------------------------------------------------
class AdjustableCombine:
    def __init__(self, coefficients):
        # Store the list of coefficients (one per weight dimension).
        assert min(coefficients) >= 0, "All coefficients must be non-negative."
        self.coefficients = coefficients

    def __call__(self, u, v, weights, feature_u=None, feature_v=None):
        # Combine the weights by computing a weighted sum.
        return sum(c * w for c, w in zip(self.coefficients, weights))
    


class test_object(STATS_OBJECT):

    def test_dual(self, dual_weight=1, n_graph = 1, n_st = 100, num_nodes=1000, weight_ranges=[(0, 10), (0, 1)], mean_n_neighbors=5., constraint_upper_bound=1):
        n_solved = 0    
        res = []
        for _ in range(n_graph):
            graph_obj = DirectedWeightedGraphNX(num_nodes, weight_ranges, mean_n_neighbors/num_nodes)
            G = graph_obj.graph
            for _ in range(n_st):
                source, target =  np.random.choice(num_nodes, size=2, replace=False)
                source, target = int(source), int(target)
                self._print(f"Source: {source}, Target: {target}", f"Dual weight: {dual_weight}", f"Graph: {G}")
                
                # Prim path
                solver = ShortestPathSolver(G, combine_func=AdjustableCombine([1, dual_weight]))
                path = solver.solve(source, target)
                if path is None:
                    self._print("No prim path found.")
                    continue
                prim_cost = solver.compute_path_weight(path, combine_func=AdjustableCombine([1, 0]))
                self._print(f"Prim path found: {path}, prim_cost: {prim_cost}")
                dual_relaxation_cost = solver.compute_path_weight(path, combine_func=AdjustableCombine([1, dual_weight]))
                self._print(f"Prim path found: {path}, dual_relaxation_cost: {dual_relaxation_cost}")
                
                prim_cost_lagrangian = dual_relaxation_cost - dual_weight * constraint_upper_bound
                self._print(f"Prim path found: {path}, prim_cost_lagrangian: {prim_cost_lagrangian}")
                
                prim_feasiblity_cost = solver.compute_path_weight(path, combine_func=AdjustableCombine([0, 1]))
                self._print(f"Prim path found: {path}, prim_feasiblity_cost: {prim_feasiblity_cost}")

                # Dual path
                solver = ShortestPathSolver(G, combine_func=AdjustableCombine([0, 1]))
                path = solver.solve(source, target)
                if path is None:
                    self._print("No dual path found.")
                    continue
                dual_cost = solver.compute_path_weight(path, combine_func=AdjustableCombine([0, 1]))
                self._print(f"Dual path found: {path}, dual_cost: {dual_cost}")
                if dual_cost > constraint_upper_bound:
                    self._print("Dual path not feasible.")
                    continue
                dual_feasiblity_cost = solver.compute_path_weight(path, combine_func=AdjustableCombine([1, 0]))
                self._print(f"Dual path found: {path}, prim_feasiblity_cost: {dual_feasiblity_cost}")

                # prim_dual_ratio = (prim_cost-prim_cost_lagrangian) / prim_cost
                if prim_feasiblity_cost <= constraint_upper_bound:
                    self._print("Prim path is feasible.")
                    dual_feasiblity_cost = prim_cost
                
                prim_dual_ratio = (dual_feasiblity_cost-prim_cost_lagrangian) / dual_feasiblity_cost
                
                self._print(f"Prim/dual ratio: {prim_dual_ratio}")
                res.append(prim_dual_ratio)
                n_solved += 1

        avg_ratio = sum(res) / max(len(res),1)
        self._printalltime(f"Dual weight: {dual_weight:2.4f}", f"Average prim/dual ratio: {avg_ratio:2.4f}", f"Percentage of solved problems: {n_solved/(n_st*n_graph):2.4f}")
        
        return avg_ratio, n_solved/(n_st*n_graph)

def main():
    test_o = test_object()
    
    pd_ratio_list = []
    sl_ratio_list = []
    
    dual_weight_lower_bound = 0.1
    dual_weight_upper_bound = 10
    num_dual_weight = 20
    dual_weight_list_log = np.logspace(np.log10(dual_weight_lower_bound), np.log10(dual_weight_upper_bound), num=num_dual_weight, endpoint=True, base=10.0, dtype=None, axis=0).tolist()
    for test_dual_weight in dual_weight_list_log:
        pd_ratio, sl_ratio =  test_o.test_dual(test_dual_weight,n_graph=1,n_st=100)
        # test_o._printalltime(f"Testing dual weight: {test_dual_weight}",f"Prim/dual ratio: {pd_ratio}, Percentage of solved problems: {sl_ratio}")
        pd_ratio_list.append(pd_ratio)
        sl_ratio_list.append(sl_ratio)
   
   
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(dual_weight_list_log, pd_ratio_list, color='skyblue')
    axs[0].set_title("Average Prim/Dual Ratio")
    
    axs[1].plot(dual_weight_list_log, sl_ratio_list, color='salmon')
    axs[1].set_title("Percentage of Solved Problems")
    
    plt.show()

if __name__ == '__main__':
    main()