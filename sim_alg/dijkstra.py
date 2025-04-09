import numpy as np
import numba
from numba import prange, typed, types

# ------------------------------------------------------------------------------
# Revised Step 1. Convert edge list to CSR representation with duplicate merging.
# ------------------------------------------------------------------------------

uni_tuple_t = types.UniTuple(types.int64, 2)
@numba.njit(cache=True)
def build_csr(num_nodes, edges, weights, merging_method='avg'):
    """
    Build CSR (Compressed Sparse Row) for an edge list, merging duplicate
    edges.
    
    Parameters:
      num_nodes: int
          The total number of nodes.
      edges: np.ndarray of shape (K, 2)
          Each row is an edge (u, v).
      weights: np.ndarray of shape (K,)
          Corresponding edge weights.
    
    Returns:
      indptr: np.ndarray of shape (num_nodes+1,)
          Index pointers into the 'indices' and 'data' arrays.
      indices: np.ndarray
          Concatenated neighbor lists.
      data: np.ndarray
          Corresponding edge weights for the neighbors.
    """
    # Use a Numba-typed dictionary to merge duplicate edges.
    merged = typed.Dict.empty(
        key_type=uni_tuple_t,
        value_type=types.float64
    )
    n_copy = typed.Dict.empty(
        key_type=uni_tuple_t,
        value_type=types.int64
    )
    K = edges.shape[0]
    # Merge duplicate edges by accumulating weights.
    for i in range(K):
        u = edges[i, 0]
        v = edges[i, 1]
        w = weights[i]
        key = (u, v)
        if key in merged:
            if merging_method == 'sum':
                merged[key] += w
            elif merging_method == 'max':
                merged[key] = max(merged[key], w)
            elif merging_method == 'min':
                merged[key] = min(merged[key], w)
            else:
                merged[key] = (merged[key] * n_copy[key] + w) / (n_copy[key]+1)   
            n_copy[key] += 1
        else:
            n_copy[key] = 1
            merged[key] = w
        key = (v, u)  # Also consider the reverse edge for undirected graphs.
        if key in merged:
            if merging_method == 'sum':
                merged[key] += w
            elif merging_method == 'max':
                merged[key] = max(merged[key], w)
            elif merging_method == 'min':
                merged[key] = min(merged[key], w)
            else:
                merged[key] = (merged[key] * n_copy[key] + w) / (n_copy[key]+1)   
            n_copy[key] += 1
        else:
            n_copy[key] = 1
            merged[key] = w

    # Create new arrays for unique edges and their accumulated weights.
    new_K = len(merged)
    new_edges = np.empty((new_K, 2), dtype=np.int64)
    new_weights = np.empty(new_K, dtype=np.float64)
    
    j = 0
    for key in merged.keys():
        new_edges[j, 0] = key[0]
        new_edges[j, 1] = key[1]
        new_weights[j] = merged[key]
        j += 1

    K = new_K  # Update K to the number of unique edges.
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    
    # Count out-degrees based on unique edges.
    for i in range(K):
        u = new_edges[i, 0]
        indptr[u+1] += 1
    
    # Compute cumulative sum for CSR indptr.
    for i in range(num_nodes):
        indptr[i+1] += indptr[i]
        
    indices = np.empty(K, dtype=np.int64)
    data = np.empty(K, dtype=new_weights.dtype)
    
    # Temporary counter to track positions.
    counter = np.zeros(num_nodes, dtype=np.int64)
    for i in range(K):
        u = new_edges[i, 0]
        pos = indptr[u] + counter[u]
        indices[pos] = new_edges[i, 1]
        data[pos] = new_weights[i]
        counter[u] += 1
    
    return indptr, indices, data

# ------------------------------------------------------------------------------
# Step 2. Custom heap for Dijkstra.
# ------------------------------------------------------------------------------
@numba.njit(cache=True)
def heap_push(heap_cost, heap_node, size, cost, node):
    heap_cost[size] = cost
    heap_node[size] = node
    pos = size
    size += 1
    # Bubble up.
    while pos > 0:
        parent = (pos - 1) // 2
        if heap_cost[parent] > heap_cost[pos]:
            tmp = heap_cost[parent]
            heap_cost[parent] = heap_cost[pos]
            heap_cost[pos] = tmp
            tmp = heap_node[parent]
            heap_node[parent] = heap_node[pos]
            heap_node[pos] = tmp
            pos = parent
        else:
            break
    return size

@numba.njit(cache=True)
def heap_pop(heap_cost, heap_node, size):
    min_cost = heap_cost[0]
    min_node = heap_node[0]
    size -= 1
    # Move the last element to the root.
    heap_cost[0] = heap_cost[size]
    heap_node[0] = heap_node[size]
    
    pos = 0
    # Bubble down.
    while True:
        left = 2 * pos + 1
        right = 2 * pos + 2
        smallest = pos
        if left < size and heap_cost[left] < heap_cost[smallest]:
            smallest = left
        if right < size and heap_cost[right] < heap_cost[smallest]:
            smallest = right
        if smallest == pos:
            break
        tmp = heap_cost[pos]
        heap_cost[pos] = heap_cost[smallest]
        heap_cost[smallest] = tmp
        tmp = heap_node[pos]
        heap_node[pos] = heap_node[smallest]
        heap_node[smallest] = tmp
        pos = smallest
    return min_cost, min_node, size

# ------------------------------------------------------------------------------
# Step 3. Single-query Dijkstra with path reconstruction.
# ------------------------------------------------------------------------------
@numba.njit(cache=True)
def single_dijkstra_with_path(num_nodes, indptr, indices, data, source, target):
    INF = 1e12  # A value larger than any realistic path cost.
    dist = np.full(num_nodes, INF)
    dist[source] = 0.0
    prev = -1 * np.ones(num_nodes, dtype=np.int64)
    
    max_heap_size = num_nodes
    heap_cost = np.empty(max_heap_size, dtype=np.float64)
    heap_node = np.empty(max_heap_size, dtype=np.int64)
    heap_size = 0
    
    heap_size = heap_push(heap_cost, heap_node, heap_size, 0.0, source)
    
    while heap_size > 0:
        current_cost, u, heap_size = heap_pop(heap_cost, heap_node, heap_size)
        if u == target:
            break
        if current_cost > dist[u]:
            continue
        for idx in range(indptr[u], indptr[u+1]):
            v = indices[idx]
            weight = data[idx]
            new_cost = current_cost + weight
            if new_cost < dist[v]:
                dist[v] = new_cost
                prev[v] = u
                heap_size = heap_push(heap_cost, heap_node, heap_size, new_cost, v)
    
    if dist[target] == INF:
        return INF, 0, -np.ones(num_nodes, dtype=np.int64)
    
    count = 0
    cur = target
    while cur != -1:
        count += 1
        cur = prev[cur]
    
    path = -np.ones(num_nodes, dtype=np.int64)
    cur = target
    for i in range(count - 1, -1, -1):
        path[i] = cur
        cur = prev[cur]
    
    return dist[target], count, path

# ------------------------------------------------------------------------------
# Step 4. Parallel processing of multiple queries.
# ------------------------------------------------------------------------------
@numba.njit(parallel=True, cache=True)
def multi_dijkstra_with_paths(num_nodes, indptr, indices, data, sources, targets):
    n_queries = sources.shape[0]
    costs = np.empty(n_queries, dtype=np.float64)
    lengths = np.empty(n_queries, dtype=np.int64)
    paths_all = -np.ones((n_queries, num_nodes), dtype=np.int64)
    
    for i in prange(n_queries):
        s = sources[i]
        t = targets[i]
        cost, length, path = single_dijkstra_with_path(num_nodes, indptr, indices, data, s, t)
        costs[i] = cost
        lengths[i] = length
        for j in range(num_nodes):
            paths_all[i, j] = path[j]
    
    return costs, lengths, paths_all

# ------------------------------------------------------------------------------
# Example usage.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example graph with duplicate edges.
    edges = np.array([
        [0, 1],
        [0, 2],
        [0, 1],  # Duplicate edge from 0 to 1.
        [1, 2],
        [1, 3],
        [2, 3],
        [2, 3]   # Duplicate edge from 2 to 3.
    ], dtype=np.int64)
    weights = np.array([2, 5, 3, 1, 4, 2, 1], dtype=np.float64)
    
    num_nodes = 7000
    # Build the CSR; duplicate edges (0,1) and (2,3) will be merged.
    indptr, indices, data = build_csr(num_nodes, edges, weights)
    
    # Multiple queries.
    sources = np.array([0, 1, 1], dtype=np.int64)
    targets = np.array([3, 3, 2], dtype=np.int64)
    
    costs, lengths, paths_all = multi_dijkstra_with_paths(num_nodes, indptr, indices, data, sources, targets)
    
    for i in range(sources.shape[0]):
        if costs[i] < 1e12:
            path = paths_all[i, :lengths[i]]
            print("Query {}: from {} to {}: cost = {}, path = {}".format(
                i, sources[i], targets[i], costs[i], path))
        else:
            print("Query {}: from {} to {} is unreachable.".format(
                i, sources[i], targets[i]))
