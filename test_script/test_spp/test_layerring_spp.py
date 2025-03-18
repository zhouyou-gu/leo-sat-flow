import numpy as np

import matplotlib.pyplot as plt

from test_script.test_spp.spp_graph import LayerRingGraph, spp_graph

from sim_src.util import STATS_OBJECT


class hop_combine:
    def __call__(self, graph, edge):
        return 1
    
class constraint_cost_only_combine:
    def __call__(self, graph, edge):
        return edge[2].get('weights')[0]

class single_weight_hop_combine:
    def __init__(self, w):
        self.w = w

    def __call__(self, graph, edge):
        return self.w * edge[2].get('weights')[0] + 1

class test_object(STATS_OBJECT):
    DEBUG_STEP = 10
    DEBUG = True
    def run(self, n_st=100, num_layers=3, num_nodes_per_layer=4, max_cost_per_hop = 2, comb_func=None):
        n_solvable = 0
        objectivef_res = []
        constraint_res = []
        sketch_ratio = []
        
        graph:spp_graph = LayerRingGraph(num_layers=num_layers, num_nodes_per_layer=num_nodes_per_layer)
        # self._print("Nodes:", graph.nodes(data=True))
        # self._print("Edges:", graph.edges(data=True))
        
        for _ in range(n_st):
            self.N_STEP += 1
            source, target = np.random.choice(num_layers * num_nodes_per_layer, size=2, replace=False)
            source, target = int(source), int(target)
            self._print(f"Source: {source}, Target: {target}, Num layers: {num_layers}, Num nodes per layer: {num_nodes_per_layer}")
            
            with graph.set_combine_func_context(hop_combine()):
                min_hop_path = graph.spp_solve(source, target)
                if min_hop_path is None:
                    self._print("No path found.")
                    continue
                min_hop = graph.count_hop_in_path(min_hop_path)
                objectivef_res.append(min_hop)
            
            with graph.set_combine_func_context(constraint_cost_only_combine()):
                min_cost_path = graph.spp_solve(source, target)
                if graph.compute_path_weight(min_cost_path) > max_cost_per_hop * min_hop:
                    self._print("Path found but cost exceeds maximum.")
                    continue
            
            n_solvable += 1
            
            with graph.set_combine_func_context(comb_func):
                dual_path = graph.spp_solve(source, target)
                objectivef_res.append(graph.count_hop_in_path(dual_path))

            with graph.set_combine_func_context(constraint_cost_only_combine()):
                dual_path_cost = graph.compute_path_weight(dual_path)
                constraint_res.append((max_cost_per_hop*min_hop)/dual_path_cost)
                
            sketch_ratio.append(graph.count_hop_in_path(dual_path)/min_hop)
                
        return n_solvable/n_st, sum(objectivef_res)/n_solvable, sum(constraint_res)/n_solvable, sum(sketch_ratio)/n_solvable
                

def main():
    path_hop_list = []
    cost_pct_list = []
    n_solvable_list = []
    sketch_ratio_list = []
    
    
    dual_weight_lower_bound = 0.1
    dual_weight_upper_bound = 10
    num_dual_weight = 20
    dual_weight_list_log = np.logspace(np.log10(dual_weight_lower_bound), np.log10(dual_weight_upper_bound), num=num_dual_weight, endpoint=True, base=10.0, dtype=None, axis=0).tolist()

    for w in dual_weight_list_log:
        res = test_object().run(n_st=1000,num_layers=10,num_nodes_per_layer=20,comb_func=single_weight_hop_combine(w))
        print(f"Number of solvable problems: {res[0]:3.4f}", f"Average objective function value: {res[1]:3.4f}", f"Average constraint value: {res[2]:3.4f}, Average sketch ratio: {res[3]:3.4f}")
        n_solvable_list.append(res[0])
        path_hop_list.append(res[1])
        cost_pct_list.append(res[2])
        sketch_ratio_list.append(res[3])

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(dual_weight_list_log, path_hop_list, color='skyblue', label='Average Objective Function Value')
    axs[0].set_title("Average Objective Function Value")
    
    
    axs[1].plot(dual_weight_list_log, n_solvable_list, color='green', label='Percentage of Solvable Problems')
    axs[1].plot(dual_weight_list_log, cost_pct_list, color='skyblue', label='Average Constraint Value')
    axs[1].plot(dual_weight_list_log, sketch_ratio_list, color='red', label='Average Sketch Ratio')

    axs[1].set_title("Average Values")
    
    
    plt.show()
    
    
if __name__ == "__main__":
    test_object.DEBUG = False
    main()