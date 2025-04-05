import numpy as np

def expand_edges_with_original(edges,repeat_per_node=4):
    # Precompute the grid of offsets with the second node's offset iterating faster
    grid = np.stack(np.meshgrid(np.arange(repeat_per_node), np.arange(repeat_per_node), indexing='ij'), axis=-1).reshape(-1, 2)
    
    # For each edge, compute the expanded edges using broadcasting:
    # Each edge is scaled by 4 and then added with every combination of offsets in grid.
    expanded_edges = (edges[:, None, :] * repeat_per_node + grid[None, :, :]).reshape(-1, 2)
    
    # Repeat each original edge 16 times so that each expanded edge corresponds to an original edge.
    repeated_original = np.repeat(edges, repeat_per_node*repeat_per_node, axis=0)
    
    return repeated_original, expanded_edges

# Example usage:
edges = np.array([[1, 2],
                  [3, 4]])  # shape (K,2)
orig_edges, new_edges = expand_edges_with_original(edges)
print("Original edges repeated:\n", orig_edges,orig_edges.shape)
print("Expanded edges:\n", new_edges,new_edges.shape)
