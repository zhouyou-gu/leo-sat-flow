import networkx as nx
import numpy as np



class ShortestPathSolver:
    def __init__(self, graph, combine_func=None):
        """
        Initialize the shortest path solver.
        
        :param graph: A NetworkX directed graph where each edge has a 'weights' attribute (a list of numbers).
                      Each node can optionally have a 'feature_vector' attribute.
        :param combine_func: A callable that takes parameters (u, v, weights, feature_u, feature_v) and returns a single cost value.
                             If None, the solver defaults to returning the first weight.
        """
        self.graph = graph
        if combine_func is None:
            self.combine_func = default_combine
        else:
            self.combine_func = combine_func



# Example usage:
if __name__ == '__main__':
    # Create a sample directed graph.
    G = nx.DiGraph()
    # Each edge has two weights.
    G.add_edge(0, 1, weights=[5, 2])
    G.add_edge(1, 2, weights=[3, 1])
    G.add_edge(0, 2, weights=[10, 5])
    G.add_edge(2, 3, weights=[2, 1])
    G.add_edge(1, 3, weights=[9, 3])
    
    # Set feature vectors for nodes (for example, using lists of numbers).
    G.nodes[0]['feature_vector'] = [1, 2]
    G.nodes[1]['feature_vector'] = [2, 3]
    G.nodes[2]['feature_vector'] = [3, 1]
    G.nodes[3]['feature_vector'] = [4, 0]
    
    # Initialize the solver with the default combination (selecting the first weight).
    solver = ShortestPathSolver(G)
    
    # Solve for the shortest path between node 0 and node 3.
    path = solver.solve(0, 3)
    print("Shortest path:", path)
    
    # Compute the aggregated weight along the obtained path using the default combine function.
    if path is not None:
        total_weight = solver.compute_path_weight(path)
        print("Aggregated weight along the path:", total_weight)
    
    # Example: Using a custom combination function that takes node feature vectors into account.
    def custom_combine(u, v, weights, feature_u, feature_v):
        # Compute the Euclidean distance between the feature vectors if available.
        if feature_u is not None and feature_v is not None:
            distance = np.linalg.norm(np.array(feature_u) - np.array(feature_v))
        else:
            distance = 0
        # Return the base weight (first element) plus the distance.
        return weights[0] + distance
    
    solver.set_combine_func(custom_combine)
    path_custom = solver.solve(0, 3)
    print("Shortest path with custom combine:", path_custom)
    if path_custom is not None:
        total_weight_custom = solver.compute_path_weight(path_custom)
        print("Aggregated weight (custom) along the path:", total_weight_custom)
    
    class WeightCombiner:
        def __init__(self, factor):
            self.factor = factor
            
        def combine(self, u, v, weights, feature_u, feature_v):
            # Example: a non-linear combination that multiplies the second weight squared by a factor,
            # then adds the first weight. (Feature vectors could also be used to adjust this calculation.)
            return weights[0] + self.factor * (weights[1] ** 2)
        
    # Create an instance of WeightCombiner.
    combiner = WeightCombiner(factor=0.5)
    
    # Pass the combine method as the combine_func.
    solver = ShortestPathSolver(G, combine_func=combiner.combine)
    path_weight_combiner = solver.solve(0, 3)
    print("Path with WeightCombiner:", path_weight_combiner, solver.compute_path_weight(path_weight_combiner))
