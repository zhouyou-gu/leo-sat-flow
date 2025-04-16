import numpy as np
import numba
from numba import prange, typed, types

uni_tuple_t = types.UniTuple(types.int64, 2)


@numba.njit(parallel=True, cache=True)
def extract_weights(A, B, none_value=0):
    """
    Numba-accelerated function that extracts weights from B based on edges in A.
    
    Parameters:
      A : (K,2) np.ndarray   - Each row represents an edge defined by (source, target)
      B : (K',3) np.ndarray  - Each row is (source, target, weight) for an edge
      
    Returns:
      weights : np.ndarray of shape (K,) where weights[i] corresponds to the weight from B
                for the edge A[i]. If an edge is not found, np.nan is returned.
    """
    # Create a typed dictionary with keys as a pair of int64 and value as float64.
    d = typed.Dict.empty(
        key_type=uni_tuple_t,
        value_type=types.float64
    )
    for i in range(B.shape[0]):
        key = (int(B[i, 0]), int(B[i, 1]))
        d[key] = B[i, 2]
    
    # Preallocate the result array.
    weights = np.empty(A.shape[0], dtype=np.float64)
    
    # Use prange to parallelize lookup over A.
    for i in prange(A.shape[0]):
        key = (int(A[i, 0]), int(A[i, 1]))
        if key in d:
            weights[i] = d[key]
        else:
            weights[i] = none_value  # You can choose a different default value if needed.
    
    return weights

@numba.njit(parallel=True, cache=True)
def csr_to_edge_list(indptr, indices, data):
    # Number of non-zero entries is given by the last element in indptr.
    nnz = indptr[-1]
    # Create an output array of shape (nnz, 3). Using float64 to hold all values.
    # If you prefer to keep the indices as an integer type, you could set up a structured array.
    out = np.empty((nnz, 3), dtype=np.float64)
    
    n_rows = indptr.shape[0] - 1  # Total number of rows in the CSR matrix.
    
    # Iterate over each row in parallel.
    for i in prange(n_rows):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        # The CSR format ensures that all non-zeros for row i
        # are stored contiguously from row_start to row_end in the 'indices' and 'data' arrays.
        for j in range(row_start, row_end):
            # First column: row index (edge source)
            out[j, 0] = i  
            # Second column: column index (edge destination)
            out[j, 1] = indices[j]
            # Third column: weight of the edge
            out[j, 2] = data[j]
            
    return out


# ------------------------------------------------------------------------------
# Revised Step 1. Convert edge list to CSR representation with duplicate merging.
# ------------------------------------------------------------------------------
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
@numba.njit(boundscheck=True, cache=True)   # enable bounds checking
def heap_push(heap_cost, heap_node, size, cost, node):
    if size >= heap_cost.size:
        raise IndexError("heap overflow")
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
    
    max_heap_size = indices.size
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

@numba.njit(parallel=True, cache=True)
def multi_dijkstra_with_paths_aw_b(num_nodes, indptr, indices, sources, targets, weights, a, b):
    n_queries = sources.shape[0]
    costs = np.empty(n_queries, dtype=np.float64)
    lengths = np.empty(n_queries, dtype=np.int64)
    paths_all = -np.ones((n_queries, num_nodes), dtype=np.int64)
    
    for i in prange(n_queries):
        s = sources[i]
        t = targets[i]
        data = weights * a[i] + b
        cost, length, path = single_dijkstra_with_path(num_nodes, indptr, indices, data, s, t)
        costs[i] = cost
        lengths[i] = length
        for j in range(num_nodes):
            paths_all[i, j] = path[j]
    
    return costs, lengths, paths_all

@numba.njit(parallel=True, cache=True)
def construct_edges_matrix_all_in_one(sources, targets, costs, lengths, paths_all):
    """
    Constructs a result matrix (with shape (total_edges, 5)) from the output of multi_dijkstra_with_paths.
    
    For each query (s-t pair) i:
      - Let L = lengths[i] (number of vertices in the returned path).
      - For every edge from the j-th vertex to the (j+1)-th vertex (j=0,..., L-2),
        a row is added with:
          Column 0 : paths_all[i, j]     (the "from" vertex for the edge)
          Column 1 : paths_all[i, j+1]   (the "to" vertex for the edge)
          Column 2 : sources[i]          (the overall source for the query)
          Column 3 : targets[i]          (the overall target for the query)
          Column 4 : costs[i]            (the s-t pair’s cost/weight)
          Column 5 : i                  (the index of the query)
          Column 6 : 0 or 1             (0 if unreachable, 1 if reachable)    
          
    The function first computes how many edges exist in total over all queries,
    by summing (lengths[i] - 1) for each query i, and also computes a prefix offset
    array so that for query i the results will be stored beginning at that offset.
    
    Parameters:
      sources   : 1D np.ndarray(int64)  of shape (Q,), each element a source vertex.
      targets   : 1D np.ndarray(int64)  of shape (Q,), each element a target vertex.
      costs     : 1D np.ndarray(float64) of shape (Q,), the s-t cost for each query.
      lengths   : 1D np.ndarray(int64)  of shape (Q,), number of vertices in each returned path.
      paths_all : 2D np.ndarray(int64)  of shape (Q, N) with the reconstructed paths (unused slots padded, e.g. with -1).
    
    Returns:
      A (total_edges x 6) np.ndarray(float64), where total_edges = sum_i (lengths[i]-1).
      Each row is [from, to, source, target, cost, idx].
    """
    Q = sources.shape[0]
    total_edges = 0
    # Build offsets array and compute total number of edges (each query contributes lengths[i]-1 edges)
    offsets = np.empty(Q, dtype=np.int64)
    for i in range(Q):
        total_edges += lengths[i] - 1
        if i == 0:
            offsets[i] = 0
        else:
            offsets[i] = offsets[i - 1] + (lengths[i - 1] - 1)
    
    # Allocate the result matrix
    result = np.empty((total_edges, 7), dtype=np.float64)
    
    # Fill the result matrix in parallel over queries
    for i in prange(Q):
        L = lengths[i]
        if L < 2:
            # No edge exists if the path has fewer than 2 vertices.
            continue
        start_idx = offsets[i]
        num_edges = L - 1
        src = sources[i]
        tgt = targets[i]
        cost = costs[i]
        for j in range(num_edges):
            result[start_idx + j, 0] = paths_all[i, j]       # from-vertex of edge j
            result[start_idx + j, 1] = paths_all[i, j + 1]   # to-vertex of edge j
            result[start_idx + j, 2] = src                   # s-t query source (same for all edges in this query)
            result[start_idx + j, 3] = tgt                   # s-t query target
            result[start_idx + j, 4] = cost                  # s-t query cost
            result[start_idx + j, 5] = i                 # s-t query index
            if cost >= 1e12:
                result[start_idx + j, 6] = 0                # unreachable
            else:
                result[start_idx + j, 6] = 1                # reachable
    return result

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
    
    data = np.tile(data, (sources.shape[0], 1)) # Repeat data for each query.
    costs, lengths, paths_all = multi_dijkstra_with_paths(num_nodes, indptr, indices, data, sources, targets)
    
    for i in range(sources.shape[0]):
        if costs[i] < 1e12:
            path = paths_all[i, :lengths[i]]
            print("Query {}: from {} to {}: cost = {}, path = {}".format(
                i, sources[i], targets[i], costs[i], path))
        else:
            print("Query {}: from {} to {} is unreachable.".format(
                i, sources[i], targets[i]))

    print("data", construct_edges_matrix_all_in_one(sources, targets, costs, lengths, paths_all)) 