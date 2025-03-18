import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def default_combine(graph, edge):
    """
    Default combine function that returns the first weight on the edge.
    :param graph: The graph object.
    :param edge: A tuple (source, target, data) representing an edge in the graph.
    :return: The first weight based on the edge.
    """
    # The feature_vector retrieval is kept commented in case it's needed in future logic.
    # feature_u = graph.nodes[edge[0]].get('feature_vector', None)
    # feature_v = graph.nodes[edge[1]].get('feature_vector', None)
    return edge[2].get('weights')[0]

def null_combine(graph, edge):
    return 0

class spp_graph(nx.DiGraph):
    DISABLE_COMBINE_CONTEXT = False
    COMBINE_CONTEXT_ON = False
    class combine_func_context:
        def __init__(self, graph, combine_func):
            self.graph = graph
            self.combine_func = combine_func
            self.active = False
        def __enter__(self):
            # Setup the context (e.g., open a connection or allocate a resource)
            self.active = True
            self.graph.COMBINE_CONTEXT_ON = True 
            self.graph._update_graph_edge_value(self.combine_func)
            # Return self (the context object) so the caller can work with it.
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            # Perform cleanup: release the resource and mark the context as inactive.
            self.graph._update_graph_edge_value(null_combine)
            self.graph.COMBINE_CONTEXT_ON = False
            self.active = False
            # Optionally, handle exceptions by returning True.
            # Here we let any exception propagate.
            # (Returning True would suppress the exception.)
            # In production code, be very deliberate about which exceptions to suppress.
            return False
        def current_combine_func(self):
            return self.combine_func.__name__ if self.active else None
        
    """
    Base class for a directed weighted graph.
    """    
    def generate_graph(self):
        """
        Abstract method to generate the graph.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses should implement generate_graph.")

    def __str__(self):
        """
        Returns a string representation of the graph.
        """
        output = []
        for source, target, data in self.edges(data=True):
            output.append(f"{source} -> {target}: {data.get('weights')}")
        return "\n".join(output)
    
    def _update_graph_edge_value(self, combine_func=null_combine):
        """
        Updates each edge in the graph with a single value computed by the combine function.
        """
        assert self.COMBINE_CONTEXT_ON or self.DISABLE_COMBINE_CONTEXT, "Combine function not set."
        nx.set_edge_attributes(self, {(edge[0], edge[1]): combine_func(self, edge)
                                       for edge in self.edges(data=True)}, 'value')
        return self 

    def spp_solve(self, source, target):
        assert self.COMBINE_CONTEXT_ON or self.DISABLE_COMBINE_CONTEXT, "Combine function not set."
        """
        Solve the shortest path problem using Dijkstra's algorithm and the current weight combination function.
        Returns only the path.
        
        :param source: The source node.
        :param target: The target node.
        :return: A list of nodes representing the shortest path, or None if no path exists.
        """
        try:
            path = nx.dijkstra_path(self, source, target, weight="value")
            return path
        except nx.NetworkXNoPath:
            return None

    def compute_path_weight(self, path):
        """
        Compute the aggregated weight along a given path using a combination function.
        
        :param path: A list of nodes representing a path in the graph.
        :param combine_func: Optional callable to combine edge weights.
        :return: The aggregated weight along the path.
        :raises ValueError: If an edge in the path does not exist in the graph.
        """
        assert self.COMBINE_CONTEXT_ON or self.DISABLE_COMBINE_CONTEXT, "Combine function not set."
        total_cost = 0
        # Iterate over each consecutive pair of nodes in the path.
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.has_edge(u, v):
                total_cost += self[u][v]['value']
            else:
                raise ValueError(f"Edge ({u}, {v}) does not exist in the graph.")
        return total_cost

    def set_combine_func_context(self, combine_func):
        """
        Set the combine function for edge weights.
        
        :param combine_func: A callable that takes parameters (u, v, weights, feature_u, feature_v) and returns a single cost value.
        """
        return self.combine_func_context(self, combine_func)
        
    def count_hop_in_path(self, path):
        return len(path) - 1

class DirectedWeightedGraphNX(spp_graph):
    def __init__(self, num_nodes, weight_ranges, extra_edge_prob=0.3):
        """
        Initialize the directed weighted graph using NetworkX.
        
        :param num_nodes: Number of nodes in the graph.
        :param weight_ranges: List of tuples where each tuple (min, max) defines the range 
                              for each weight dimension on an edge.
        :param extra_edge_prob: Probability of adding an extra edge between two nodes.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.weight_ranges = weight_ranges
        self.extra_edge_prob = extra_edge_prob
        self.generate_graph()

    def generate_graph(self):
        """
        Generates the graph by first ensuring strong connectivity using a spanning tree
        and then adding extra random edges based on the specified probability.
        """
        # Create a spanning tree to ensure connectivity.
        for i in range(1, self.num_nodes):
            j = random.randint(0, i-1)
            self._add_edge(j, i)

        # Optionally add extra edges.
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and not self.has_edge(i, j):
                    if random.random() < self.extra_edge_prob:
                        self._add_edge(i, j)

    def _add_edge(self, source, target):
        """
        Adds a directed edge from source to target with randomly generated weights.
        """
        # Debug output; replace with proper logging if needed.
        # print("Adding edge:", source, "->", target)
        if not self.has_edge(source, target):
            self.add_edge(source, target, weights=self.generate_random_weights())

    def generate_random_weights(self):
        """
        Generates a list of random weights for an edge, one for each weight dimension.
        """
        return [random.uniform(r_min, r_max) for (r_min, r_max) in self.weight_ranges]

class LayerRingGraph(spp_graph):
    def __init__(self, num_layers, num_nodes_per_layer, weight_in_layers=None, weight_between_layers = 1.):
        """
        Initialize the layered ring graph.
        
        :param num_layers: Number of layers.
        :param num_nodes_per_layer: Number of nodes per layer.
        :param weight_in_layers: List of weights for intra-layer edges.
        :param weight_between_layers: Matrix (list of lists) defining weights for inter-layer edges.
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_nodes_per_layer = num_nodes_per_layer
        if weight_in_layers is None:
            weight_in_layers = [i*2+1 for i in range(num_layers)]
        
        assert len(weight_in_layers) == num_layers, "Number of layers and weights do not match."
        assert isinstance(weight_between_layers, (int, float)), "Inter-layer weights must be a single value."
        self.weight_in_layers = weight_in_layers
        self.weight_between_layers = weight_between_layers
        self.generate_graph()

    def generate_graph(self):
        """
        Generates a ring graph with layers and inter-layer connections.
        """
        for i in range(self.num_layers):
            for j in range(self.num_nodes_per_layer):
                node_id = i * self.num_nodes_per_layer + j
                self.add_node(node_id, loc=(i, j))
                if j > 0:
                    self._add_edge(node_id - 1, node_id, i, i)
                    self._add_edge(node_id, node_id - 1, i, i)
                if i > 0:
                    self._add_edge((i - 1) * self.num_nodes_per_layer + j, node_id, i - 1, i)
                    self._add_edge(node_id, (i - 1) * self.num_nodes_per_layer + j, i, i - 1)
            # Complete the ring within the layer
            self._add_edge(i * self.num_nodes_per_layer, i * self.num_nodes_per_layer + self.num_nodes_per_layer - 1, i, i)
            self._add_edge(i * self.num_nodes_per_layer + self.num_nodes_per_layer - 1, i * self.num_nodes_per_layer, i, i)
    
    def _add_edge(self, source, target, source_layer, target_layer):
        """
        Adds a directed edge between two nodes with predefined weights.
        """
        # Debug output; replace with logging if necessary.
        # print("LayerRingGraph edge:", source, "->", target, "from layer", source_layer, "to", target_layer)
        if not self.has_edge(source, target):
            if source_layer == target_layer:
                self.add_edge(source, target, weights=[self.weight_in_layers[source_layer]])
            else:
                self.add_edge(source, target, weights=[self.weight_between_layers])

# Example usage:
if __name__ == '__main__':
    # Example 1: DirectedWeightedGraphNX
    print("DirectedWeightedGraphNX Example:")
    G1 = DirectedWeightedGraphNX(num_nodes=5, weight_ranges=[(1, 10), (0, 1)], extra_edge_prob=0.4)
    print(G1)
    
    # Example 2: LayerRingGraph
    print("\nLayerRingGraph Example:")
    num_layers = 10
    num_nodes_per_layer = 10
    G2 = LayerRingGraph(num_layers, num_nodes_per_layer)
    print(G2)
    if True:
        G2.DISABLE_COMBINE_CONTEXT=True
        G2._update_graph_edge_value(default_combine)
        # Plotting the LayerRingGraph
        weights = list(nx.get_edge_attributes(G2, 'value', 0).values())
        if weights:
            norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
            cmap = plt.cm.Blues
            edge_colors = [cmap(norm(weight)) for weight in weights]
        else:
            edge_colors = None

        pos = nx.spring_layout(G2, iterations=1000)
        nx.draw_networkx_nodes(G2, pos, node_color='lightblue', node_size=100)
        nx.draw_networkx_edges(G2, pos, edge_color=edge_colors, width=1, arrows=False)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Edge Weight', ax=plt.gca())
        
        plt.title("LayerRingGraph Visualization")
        plt.show()
