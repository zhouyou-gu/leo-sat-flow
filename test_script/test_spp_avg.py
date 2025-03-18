import random
import networkx as nx
import numpy as np
import pulp
import matplotlib.pyplot as plt
import time

# ------------------------------
# Graph Generation
# ------------------------------
def generate_random_graph(n, extra_edges):
    """
    Generates a directed graph with n nodes.
    A spanning tree is created to ensure connectivity from node 0 (source) to node n-1 (target),
    then extra_edges random directed edges are added.
    Each edge gets a random delay (1 to 10) and an error rate (uniform between 0.001 and 0.01).
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    # Create a spanning tree.
    for i in range(1, n):
        j = random.randint(0, i-1)
        G.add_edge(j, i, delay=random.uniform(0.1, 1), error=random.uniform(0.1, 1))
    
    # Add extra random edges.
    added = 0
    while added < extra_edges:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, delay=random.uniform(0.1, 1), error=random.uniform(0.1, 1)-0.2)
            added += 1
    return G

# ------------------------------
# MILP and LP Relaxation Solvers
# ------------------------------
def solve_milp(G, source, target, T, time_limit=60):
    """
    Solve the constrained MILP:
      min ∑ delay_ij * x_ij
      s.t. flow conservation,
           ∑ error_ij * x_ij ≤ T,
           x_ij ∈ {0,1}.
    Returns (p_star, status, solution dictionary).
    """
    prob = pulp.LpProblem("MinDelayPath_with_ErrorConstraint", pulp.LpMinimize)
    x = {}
    for (i, j) in G.edges():
        x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
    
    prob += pulp.lpSum(G[i][j]['delay'] * x[(i, j)] for (i, j) in G.edges()), "Total_Delay"
    
    for node in G.nodes():
        inflow = pulp.lpSum(x[(i, j)] for (i, j) in G.in_edges(node))
        outflow = pulp.lpSum(x[(i, j)] for (i, j) in G.out_edges(node))
        if node == source:
            prob += (outflow - inflow == 1), f"Flow_{node}"
        elif node == target:
            prob += (inflow - outflow == 1), f"Flow_{node}"
        else:
            prob += (outflow - inflow == 0), f"Flow_{node}"
    
    prob += pulp.lpSum(G[i][j]['error'] * x[(i, j)] for (i, j) in G.edges()) <= T, "Error_Constraint"
    
    solver = pulp.GLPK_CMD(timeLimit=time_limit, msg=0)
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    if status == "Optimal":
        p_star = pulp.value(prob.objective)
        solution = {edge: pulp.value(var) for edge, var in x.items()}
    else:
        p_star = None
        solution = None
    return p_star, status, solution

def solve_lp_relaxation(G, source, target, T, time_limit=60):
    """
    Solve the LP relaxation of the MILP (x continuous in [0,1]).
    Returns the LP optimum value.
    """
    prob = pulp.LpProblem("MinDelayPath_with_ErrorConstraint_LP", pulp.LpMinimize)
    x = {}
    for (i, j) in G.edges():
        x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat='Continuous')
    
    prob += pulp.lpSum(G[i][j]['delay'] * x[(i, j)] for (i, j) in G.edges()), "Total_Delay"
    
    for node in G.nodes():
        inflow = pulp.lpSum(x[(i, j)] for (i, j) in G.in_edges(node))
        outflow = pulp.lpSum(x[(i, j)] for (i, j) in G.out_edges(node))
        if node == source:
            prob += (outflow - inflow == 1), f"Flow_{node}"
        elif node == target:
            prob += (inflow - outflow == 1), f"Flow_{node}"
        else:
            prob += (outflow - inflow == 0), f"Flow_{node}"
    
    prob += pulp.lpSum(G[i][j]['error'] * x[(i, j)] for (i, j) in G.edges()) <= T, "Error_Constraint"
    
    solver = pulp.GLPK_CMD(timeLimit=time_limit, msg=0)
    prob.solve(solver)
    p_lp = pulp.value(prob.objective)
    return p_lp

# ------------------------------
# Single Dual Relaxation
# ------------------------------
def solve_single_dual_bound(G, source, target, T, lambda_grid):
    """
    For each lambda in lambda_grid, modify edge cost:
         cost = delay + lambda * error.
    Compute the shortest path and dual function value:
         g_single(lambda) = (shortest path cost) - lambda*T.
    Returns the best dual bound, best lambda, and corresponding path.
    """
    best_bound = -float('inf')
    best_lambda = None
    best_path = None
    for lam in lambda_grid:
        for u, v, data in G.edges(data=True):
            data['modified_cost'] = data['delay'] + lam * data['error']
        try:
            sp_cost = nx.shortest_path_length(G, source=source, target=target, weight='modified_cost',method='bellman-ford')
            path = nx.shortest_path(G, source=source, target=target, weight='modified_cost',method='bellman-ford')
        except nx.NetworkXNoPath:
            sp_cost = float('inf')
            path = None
        dual_val = sp_cost - lam * T
        if dual_val > best_bound:
            best_bound = dual_val
            best_lambda = lam
            best_path = path
    return best_bound, best_lambda, best_path

# ------------------------------
# Multi-Dual Variable Optimization (Subgradient)
# ------------------------------
def unconstrained_shortest_path(G, source, target, lambda_dict):
    """
    Given dual variables (lambda_dict) on each edge, compute the shortest path with modified cost:
         c_ij = delay + lambda_ij * error.
    Returns the cost and the corresponding path.
    """
    for u, v, data in G.edges(data=True):
        lam = lambda_dict.get((u, v), 0)
        data['mod_cost'] = data['delay'] + lam * data['error']
    try:
        cost = nx.shortest_path_length(G, source=source, target=target, weight='mod_cost',method='bellman-ford')
        path = nx.shortest_path(G, source=source, target=target, weight='mod_cost',method='bellman-ford')
    except nx.NetworkXNoPath:
        cost = float('inf')
        path = None
    return cost, path

def multi_dual_optimization(G, source, target, T, num_iters=50, step_size=0.01):
    """
    Optimize dual variables (one per edge) via subgradient ascent.
    For current duals lambda_dict, the dual function is:
         g(lambda) = SP(c(lambda)) - (max lambda)*T,
    where SP(c(lambda)) is the unconstrained shortest path cost with modified costs.
    Returns the best dual value, best lambda_dict, and corresponding path.
    """
    # Initialize dual variables to zero.
    lambda_dict = {(u, v): 0.0 for u, v in G.edges()}
    best_dual = -float('inf')
    best_lambda = lambda_dict.copy()
    best_path = None
    
    for it in range(num_iters):
        sp_cost, sp_path = unconstrained_shortest_path(G, source, target, lambda_dict)
        mu = max(lambda_dict.values()) if lambda_dict else 0
        dual_val = sp_cost - mu * T
        
        if dual_val > best_dual:
            best_dual = dual_val
            best_lambda = lambda_dict.copy()
            best_path = sp_path
        
        # Create an indicator for edges on the current shortest path.
        x_star = {(u, v): 1 if best_path and (u, v) in list(zip(best_path, best_path[1:])) else 0 for u, v in G.edges()}
        
        for (u, v) in G.edges():
            subgrad = G[u][v]['error'] * x_star[(u, v)]
            lambda_dict[(u, v)] += step_size * subgrad
            if lambda_dict[(u, v)] < 0:
                lambda_dict[(u, v)] = 0.0
                
    return best_dual, best_lambda, best_path

# ------------------------------
# Helper Functions: Path Extraction & Metrics
# ------------------------------
def extract_path(G, source, target, solution):
    """
    Given a solution dictionary (edge: value), extract a path from source to target.
    """
    selected_edges = [edge for edge, val in solution.items() if val > 0.5]
    H = nx.DiGraph()
    H.add_edges_from(selected_edges)
    try:
        path = nx.shortest_path(H, source=source, target=target)
    except nx.NetworkXNoPath:
        path = None
    return path

def compute_path_delay(G, path):
    """
    Compute the total delay along a given path.
    """
    if path is None:
        return None
    total_delay = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        total_delay += G[u][v]['delay']
    return total_delay

def compute_path_error(G, path):
    """
    Compute the cumulative error along a given path.
    """
    if path is None:
        return None
    total_error = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        total_error += G[u][v]['error']
    return total_error

# ------------------------------
# Experiment Runner and Plotting
# ------------------------------
def run_experiment(n=1000, extra_edges=40000, T=0,
                   lambda_min=0, lambda_max=100, num_lambda=100,
                   num_trials=10, dual_iters=50, step_size=0.01):
    """
    Runs num_trials experiments. For each trial:
      - Generate a random graph.
      - Solve MILP and LP relaxation.
      - Compute single dual relaxation (via grid search).
      - Compute multi dual optimization (via subgradient method).
      - Extract paths and compute delay & error.
      - Compute duality gaps.
    Returns lists for:
      - gap_lp: (MILP - LP)
      - gap_single: (MILP - single dual)
      - gap_multi: (MILP - multi dual)
      - MILP delay & error,
      - Single dual delay & error,
      - Multi dual delay & error.
    """
    gap_lp_list = []
    gap_single_list = []
    gap_multi_list = []
    milp_delay_list = []
    milp_error_list = []
    single_delay_list = []
    single_error_list = []
    multi_delay_list = []
    multi_error_list = []
    
    for trial in range(num_trials):
        print(f"Trial {trial+1}/{num_trials}")
        G = generate_random_graph(n, extra_edges)
        source = 0
        target = n - 1
        
        # MILP.
        p_star, status, milp_solution = solve_milp(G, source, target, T)
        if p_star is None:
            print("  MILP infeasible. Skipping trial.\n")
            continue
        milp_path = extract_path(G, source, target, milp_solution)
        milp_delay = compute_path_delay(G, milp_path)
        milp_error = compute_path_error(G, milp_path)
        
        # LP relaxation.
        p_lp = solve_lp_relaxation(G, source, target, T)
        
        # Single dual relaxation.
        lambda_grid = np.linspace(lambda_min, lambda_max, num_lambda)
        g_single, best_lambda_val, single_path = solve_single_dual_bound(G, source, target, T, lambda_grid)
        single_delay = compute_path_delay(G, single_path)
        single_error = compute_path_error(G, single_path)
        
        # Multi dual optimization.
        g_multi, best_lambda_dict, multi_path = multi_dual_optimization(G, source, target, T, num_iters=dual_iters, step_size=step_size)
        multi_delay = compute_path_delay(G, multi_path)
        multi_error = compute_path_error(G, multi_path)
        
        # Compute gaps.
        gap_lp = p_star - p_lp
        gap_single = p_star - g_single
        gap_multi = p_star - g_multi
        
        print(f"  MILP Value: {p_star:.2f}")
        print(f"  LP Relaxation: {p_lp:.2f}")
        print(f"  Single Dual Bound: {g_single:.2f} (λ={best_lambda_val:.2f})")
        print(f"  Multi Dual Bound: {g_multi:.2f}")
        print(f"  Gaps: MILP-LP: {gap_lp:.2f}, MILP-Single: {gap_single:.2f}, MILP-Multi: {gap_multi:.2f}")
        print(f"  MILP Path Delay: {milp_delay:.2f}, Error: {milp_error:.4f}")
        print(f"  Single Dual Path Delay: {single_delay:.2f}, Error: {single_error:.4f}")
        print(f"  Multi Dual Path Delay: {multi_delay:.2f}, Error: {multi_error:.4f}\n")
        
        gap_lp_list.append(gap_lp)
        gap_single_list.append(gap_single)
        gap_multi_list.append(gap_multi)
        milp_delay_list.append(milp_delay if milp_delay is not None else 0)
        milp_error_list.append(milp_error if milp_error is not None else 0)
        single_delay_list.append(single_delay if single_delay is not None else 0)
        single_error_list.append(single_error if single_error is not None else 0)
        multi_delay_list.append(multi_delay if multi_delay is not None else 0)
        multi_error_list.append(multi_error if multi_error is not None else 0)
    
    return (gap_lp_list, gap_single_list, gap_multi_list,
            milp_delay_list, milp_error_list,
            single_delay_list, single_error_list,
            multi_delay_list, multi_error_list)

if __name__ == "__main__":
    num_trials = 10  # Adjust as desired.
    (gap_lp_list, gap_single_list, gap_multi_list,
     milp_delay_list, milp_error_list,
     single_delay_list, single_error_list,
     multi_delay_list, multi_error_list) = run_experiment(num_trials=num_trials)
    
    trials = np.arange(1, len(gap_lp_list) + 1)
    
    # Plotting: 7 subplots.
    fig, axs = plt.subplots(7, 1, figsize=(5, 5))
    
    # Subplot 1: Duality Gaps.
    axs[0].plot(trials, gap_lp_list, 'x-', label='MILP - LP')
    axs[0].plot(trials, gap_single_list, '+-', label='MILP - Single Dual')
    axs[0].plot(trials, gap_multi_list, 's-', label='MILP - Multi Dual')
    axs[0].set_xlabel("Trial")
    axs[0].set_ylabel("Duality Gap")
    axs[0].set_title("Duality Gaps")
    axs[0].legend()
    axs[0].grid(True)
    
    # Subplot 2: MILP Delay.
    axs[1].plot(trials, milp_delay_list, 'o-', color='green')
    axs[1].set_xlabel("Trial")
    axs[1].set_ylabel("MILP Delay")
    axs[1].set_title("MILP Optimal Delay")
    axs[1].grid(True)
    
    # Subplot 3: MILP Path Error.
    axs[2].plot(trials, milp_error_list, 's-', color='red')
    axs[2].set_xlabel("Trial")
    axs[2].set_ylabel("MILP Path Error")
    axs[2].set_title("MILP Path Error")
    axs[2].grid(True)
    
    # Subplot 4: Single Dual Delay.
    axs[3].plot(trials, single_delay_list, 'd-', color='purple')
    axs[3].set_xlabel("Trial")
    axs[3].set_ylabel("Single Dual Delay")
    axs[3].set_title("Single Dual Relaxation Path Delay")
    axs[3].grid(True)
    
    # Subplot 5: Single Dual Path Error.
    axs[4].plot(trials, single_error_list, 'p-', color='orange')
    axs[4].set_xlabel("Trial")
    axs[4].set_ylabel("Single Dual Path Error")
    axs[4].set_title("Single Dual Relaxation Path Error")
    axs[4].grid(True)
    
    # Subplot 6: Multi Dual Delay.
    axs[5].plot(trials, multi_delay_list, 'v-', color='brown')
    axs[5].set_xlabel("Trial")
    axs[5].set_ylabel("Multi Dual Delay")
    axs[5].set_title("Multi Dual Relaxation Path Delay")
    axs[5].grid(True)
    
    # Subplot 7: Multi Dual Path Error.
    axs[6].plot(trials, multi_error_list, '^-', color='cyan')
    axs[6].set_xlabel("Trial")
    axs[6].set_ylabel("Multi Dual Path Error")
    axs[6].set_title("Multi Dual Relaxation Path Error")
    axs[6].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Average Gap (MILP - LP):", np.mean(gap_lp_list))
    print("Average Gap (MILP - Single Dual):", np.mean(gap_single_list))
    print("Average Gap (MILP - Multi Dual):", np.mean(gap_multi_list))
    print("Average MILP Delay:", np.mean(milp_delay_list))
    print("Average MILP Path Error:", np.mean(milp_error_list))
    print("Average Single Dual Delay:", np.mean(single_delay_list))
    print("Average Single Dual Path Error:", np.mean(single_error_list))
    print("Average Multi Dual Delay:", np.mean(multi_delay_list))
    print("Average Multi Dual Path Error:", np.mean(multi_error_list))
