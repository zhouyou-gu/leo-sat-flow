import numpy as np
from scipy.optimize import minimize

from spp_graph import *
from test_script.test_spp.z_spp_solver import *

class SimpleGraph(spp_graph):
    def __init__(self):
        """
        Initialize a simple graph with 3 nodes and 2 edges.
        """
        self.graph = nx.DiGraph()
        self.graph.add_edge(0, 1, weights=[1, 1, 1])
        self.graph.add_edge(1, 0, weights=[1, 1, 1])
        self.graph.add_edge(1, 2, weights=[1, 1, 1])
        self.graph.add_edge(2, 1, weights=[1, 1, 1])
        self.graph.add_edge(2, 3, weights=[1, 10, 1])
        self.graph.add_edge(3, 4, weights=[1, 10, 1])

class EdgeWeightCombine:
    def __init__(self):
        # Store the list of coefficients (one per weight dimension).
        pass

    def __call__(self, weights):
        # Combine the weights by computing a weighted sum.
        return weights[0] + weights[2] * weights[1]
       
class SingleWeight:
    """
    Example class that computes a concave function f(x) = sqrt(x[0]) + sqrt(x[1]).
    In practice, you would replace the `evaluate` method with your own black-box logic.
    """
    
    def __init__(self, s, t):
        # You could store any relevant data inside this class if needed.
        self.s = s
        self.t = t
        self.graph = SimpleGraph().graph
    
    def evaluate(self, x):
        """
        Computes the concave objective value f(x).
        This function is presumably expensive or complex in your real application.
        """
        solver = ShortestPathSolver(self.graph, EdgeWeightCombine())
        path = solver.solve(self.s, self.t)
        prim_cost = solver.compute_path_weight(path, combine_func=EdgeWeightCombine())
        return prim_cost

def solve_concave_problem(x0, func_obj, max_iter=1000):
    """
    Uses COBYLA to solve:
       minimize   -f(x)
       subject to x[i] >= 0 for all i
    starting from an initial guess x0.
    `func_obj` is an instance of a class that must have a `.evaluate(x)` method returning f(x).
    """
    n = len(x0)
    
    # Negative wrapper for the class-based objective:
    def negative_f(x):
        return -func_obj.evaluate(x)

    # Build constraints list: x[i] >= 0
    constraints = []
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i]
        })
    
    # Run COBYLA
    result = minimize(
        fun=negative_f,            # objective to minimize
        x0=x0,                     # initial guess
        method='COBYLA',
        constraints=constraints,
        options={'maxiter': max_iter, 'disp': True}
    )
    return result


if __name__ == "__main__":
    # Create an instance of your black-box function class:
    my_function_obj = SimpleGraph(0,1)
    # Initial guess:
    x0 = np.array([1.0])
    # Solve:
    result = solve_concave_problem(x0, my_function_obj)
    # Print results:
    print("Optimization Result:")
    print("  Status :", result.message)
    print("  x*     :", result.x)
    print("  f(x*)  :", my_function_obj.evaluate(result.x))
