from numba import njit, prange
import numpy as np
import math



@njit(parallel=True,cache=True)
def lat_lon_to_xyz(lat_lon, r=1.):
    """
    Convert latitude and longitude arrays (in radians) to 3D Cartesian coordinates on a sphere.

    Parameters
    ----------
    lat_lon : np.ndarray
        Array of shape (n, 2) where each row is [latitude, longitude] in radians.
    r : float, optional
        Radius of the sphere (default is 1.).

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) with Cartesian coordinates [x, y, z].
                
    """
    n = lat_lon.shape[0]
    xyz = np.empty((n, 3), dtype=lat_lon.dtype)
    
    for i in prange(n):
        lat = lat_lon[i, 0]
        lon = lat_lon[i, 1]
        xyz[i, 0] = r * math.cos(lat) * math.cos(lon)  # x
        xyz[i, 1] = r * math.cos(lat) * math.sin(lon)  # y
        xyz[i, 2] = r * math.sin(lat)  # z

    return xyz

@njit(parallel=True,cache=True)
def xyz_to_lat_lon(xyz: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (lat, lon, r).

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (n, 3) with Cartesian coordinates [x, y, z].
    Returns
    -------
    np.ndarray
        Array of shape (n, 2) with spherical coordinates [latitude, longitude] in radians.
    """
    n = xyz.shape[0]
    lat_lon = np.empty((n, 2), dtype=xyz.dtype)
    
    for i in prange(n):
        x = xyz[i, 0]
        y = xyz[i, 1]
        z = xyz[i, 2]
        r = math.sqrt(x * x + y * y + z * z)
        if r == 0:
            lat_lon[i, 0] = 0.0
            lat_lon[i, 1] = 0.0
        else:
            x /= r
            y /= r
            z /= r
            # Compute latitude and longitude
            lat_lon[i, 0] = math.asin(z)  # latitude
            lat_lon[i, 1] = math.atan2(y, x)  # longitude
    return lat_lon

@njit(parallel=True,cache=True)
def rotate_deg_in_vector(vectors: np.ndarray, angle_deg: float, anglar_vector: np.ndarray = np.array([0,0,1])) -> np.ndarray:
    '''
    Rotate vectors by given angles around a specified angular vector.
    Parameters
    ----------
    vectors : np.ndarray
        Array of shape (n, 3) containing vectors to be rotated.
    angle_deg : float
        Angle in degrees by which to rotate the vectors.
    anglar_vector : np.ndarray
        Angular vector of shape (3,) around which to rotate the vectors.
    Returns
    -------
    np.ndarray
        Array of shape (n, 3) containing the rotated vectors.
    '''
    n = vectors.shape[0]
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    anglar_vector_norm = anglar_vector[0]**2 + anglar_vector[1]**2 + anglar_vector[2]**2
    anglar_vector_norm = math.sqrt(anglar_vector_norm)
    if anglar_vector_norm == 0:
        raise ValueError("Angular vector cannot be zero vector.")
    anglar_vector = anglar_vector / anglar_vector_norm  # Normalize the angular vector
    ux, uy, uz = anglar_vector
    rotated_vectors = np.empty_like(vectors)
    for i in prange(n):
        x, y, z = vectors[i]
        # Apply Rodrigues' rotation formula
        rotated_vectors[i, 0] = (x * (cos_angle + ux * ux * (1 - cos_angle)) +
                                    y * (ux * uy * (1 - cos_angle) - uz * sin_angle) + 
                                    z * (ux * uz * (1 - cos_angle) + uy * sin_angle))
        rotated_vectors[i, 1] = (x * (uy * ux * (1 - cos_angle) + uz * sin_angle) +
                                    y * (cos_angle + uy * uy * (1 - cos_angle)) +
                                    z * (uy * uz * (1 - cos_angle) - ux * sin_angle))
        rotated_vectors[i, 2] = (x * (uz * ux * (1 - cos_angle) - uy * sin_angle) +
                                    y * (uz * uy * (1 - cos_angle) + ux * sin_angle) +
                                    z * (cos_angle + uz * uz * (1 - cos_angle)))
    # Return the rotated vectors
    return rotated_vectors

@njit(parallel=True,cache=True)
def rotate_deg_in_vector_element_wise(vectors: np.ndarray, angles_degs: np.ndarray, angular_vectors: np.ndarray) -> np.ndarray:
    '''
    Rotate vectors by given angles around a specified angular vector.
    
    Parameters
    ----------
    vectors : np.ndarray
        Array of shape (n, 3) containing vectors to be rotated.
    angles_deg : np.ndarray
        Array of shape (n,) containing angles in degrees by which to rotate the vectors.
    angular_vectors : np.ndarray
        Angular vector of shape (n, 3) around which to rotate the vectors.

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) containing the rotated vectors.
    '''
    n = vectors.shape[0]
    rotated_vectors = np.empty_like(vectors)

    for i in prange(n):
        x, y, z = vectors[i]
        cos_angle = np.cos(np.deg2rad(angles_degs[i]))
        sin_angle = np.sin(np.deg2rad(angles_degs[i]))
        angular_vector_norm = angular_vectors[i, 0]**2 + angular_vectors[i, 1]**2 + angular_vectors[i, 2]**2
        angular_vector_norm = math.sqrt(angular_vector_norm)
        ux, uy, uz = angular_vectors[i]/ angular_vector_norm  # Normalize the angular vector
        rotated_vectors[i, 0] = (x * (cos_angle + ux * ux * (1 - cos_angle)) +
                                    y * (ux * uy * (1 - cos_angle) - uz * sin_angle) +
                                    z * (ux * uz * (1 - cos_angle) + uy * sin_angle))
        rotated_vectors[i, 1] = (x * (uy * ux * (1 - cos_angle) + uz * sin_angle) +
                                    y * (cos_angle + uy * uy * (1 - cos_angle)) +
                                    z * (uy * uz * (1 - cos_angle) - ux * sin_angle))
        rotated_vectors[i, 2] = (x * (uz * ux * (1 - cos_angle) - uy * sin_angle) +
                                    y * (uz * uy * (1 - cos_angle) + ux * sin_angle) +
                                    z * (cos_angle + uz * uz * (1 - cos_angle)))
    return rotated_vectors

@njit(parallel=True,cache=True)
def rotation_matmul(A: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """
    Perform parallel matrix multiplication of an (k x 3) matrix A with a (3 x 3) matrix B.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (k, 3).
    B : np.ndarray
        Input matrix of shape (3, 3).

    Returns
    -------
    np.ndarray
        Output matrix of shape (k, 3), result of A @ B.
    """
    # Ensure input dimensions
    k, n = A.shape
    if n != 3 or rot.shape != (3, 3):
        raise ValueError("A must have shape (k,3) and B must have shape (3,3)")

    # Allocate output
    C = np.empty((k, 3), dtype=A.dtype)

    # Parallel loop over rows of A
    for i in prange(k):
        # Compute dot product of A[i, :] with each column of B
        for j in range(3):
            tmp = 0.0
            for l in range(3):
                tmp += A[i, l] * rot[l, j]
            C[i, j] = tmp
    return C

@njit(parallel=True,cache=True)
def update_arrows(velocities, positions):
    """
    Compute the direction arrows in parallel.

    Parameters:
    -----------
    velocities : np.ndarray
        Array of shape (n, 3) containing velocity vectors.
    positions : np.ndarray
        Array of shape (n, 3) containing position vectors.

    Returns:
    --------
    front : np.ndarray
        Normalized velocity vectors.
    back : np.ndarray
        Negated front vectors.
    down : np.ndarray
        Normalized position vectors.
    right : np.ndarray
        Cross product of down and front vectors.
    left : np.ndarray
        Negated right vectors.
    """
    n = velocities.shape[0]
    front = np.empty_like(velocities)
    back = np.empty_like(velocities)
    down = np.empty_like(positions)
    right = np.empty_like(positions)
    left = np.empty_like(positions)

    for i in prange(n):
        # Normalize the velocity vector for the "front" arrow
        v0 = velocities[i, 0]
        v1 = velocities[i, 1]
        v2 = velocities[i, 2]
        norm_v = math.sqrt(v0*v0 + v1*v1 + v2*v2)
        front[i, 0] = v0 / norm_v
        front[i, 1] = v1 / norm_v
        front[i, 2] = v2 / norm_v

        # "Back" is simply the negative of "front"
        back[i, 0] = -front[i, 0]
        back[i, 1] = -front[i, 1]
        back[i, 2] = -front[i, 2]

        # Normalize the position vector for the "down" arrow
        p0 = -positions[i, 0]
        p1 = -positions[i, 1]
        p2 = -positions[i, 2]
        norm_p = math.sqrt(p0*p0 + p1*p1 + p2*p2)
        down[i, 0] = p0 / norm_p
        down[i, 1] = p1 / norm_p
        down[i, 2] = p2 / norm_p

        # Compute the cross product for "right": cross(down, front)
        right[i, 0] = down[i, 1] * front[i, 2] - down[i, 2] * front[i, 1]
        right[i, 1] = down[i, 2] * front[i, 0] - down[i, 0] * front[i, 2]
        right[i, 2] = down[i, 0] * front[i, 1] - down[i, 1] * front[i, 0]

        # "Left" is simply the negative of "right"
        left[i, 0] = -right[i, 0]
        left[i, 1] = -right[i, 1]
        left[i, 2] = -right[i, 2]

    return front, back, down, right, left

@njit(parallel=True,cache=True)
def filter_and_compute_pair(edges, view_from_stack, view_to_stack, cos_threshold):
    """
    This function replicates the following operations:
    
      1. Create boolean arrays (i_j_indicator and j_i_indicator) by comparing each element 
         of view_from_stack and view_to_stack to cos_threshold.
      2. For each row, use a manual "any" to compute binary indicators.
      3. Filter rows (and corresponding edges) where both binary indicators are True.
      4. For the filtered rows, compute the broadcasted pairwise boolean AND between 
         view_from_stack and view_to_stack indicators and flatten the result.
    
    Parameters
    ----------
    edges : np.ndarray
        Array of shape (n, m_edges) representing edge data (e.g., indices).
    view_from_stack : np.ndarray
        Array of shape (n, m1) with float values.
    view_to_stack : np.ndarray
        Array of shape (n, m2) with float values.
    cos_threshold : float
        Threshold for comparison.
        
    Returns
    -------
    filtered_edges : np.ndarray
        Filtered rows of edges.
    filtered_view_from_stack : np.ndarray
        Filtered rows of view_from_stack.
    filtered_view_to_stack : np.ndarray
        Filtered rows of view_to_stack.
    p_lisl_LT_pair : np.ndarray
        Flattened boolean array from the broadcasted pairwise AND between 
        the filtered view_from_stack and view_to_stack boolean indicators.
    """
    n = view_from_stack.shape[0]
    m1 = view_from_stack.shape[1]
    m2 = view_to_stack.shape[1]
    m_edges = edges.shape[1]
    
    # Step 1: Compute boolean indicator arrays.
    i_j_indicator = np.empty((n, m1), dtype=np.bool_)
    j_i_indicator = np.empty((n, m2), dtype=np.bool_)
    for i in prange(n):
        for j in range(m1):
            i_j_indicator[i, j] = view_from_stack[i, j] > cos_threshold
        for j in range(m2):
            j_i_indicator[i, j] = view_to_stack[i, j] > cos_threshold
    
    # Step 2: Compute per-row "any" (binary indicators).
    i_j_binary = np.empty(n, dtype=np.bool_)
    j_i_binary = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        flag_from = False
        for j in range(m1):
            if i_j_indicator[i, j]:
                flag_from = True
                break
        i_j_binary[i] = flag_from

        flag_to = False
        for j in range(m2):
            if j_i_indicator[i, j]:
                flag_to = True
                break
        j_i_binary[i] = flag_to
    
    # Step 3: Final indicator: only rows where both are True.
    final_indicator = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        final_indicator[i] = i_j_binary[i] and j_i_binary[i]
    
    # Count rows passing the final condition.
    count = 0
    for i in range(n):
        if final_indicator[i]:
            count += 1
            
    # Allocate filtered arrays.
    filtered_edges = np.empty((count, m_edges), dtype=edges.dtype)
    filtered_view_from_stack = np.empty((count, m1), dtype=view_from_stack.dtype)
    filtered_view_to_stack = np.empty((count, m2), dtype=view_to_stack.dtype)
    filtered_i_j_indicator = np.empty((count, m1), dtype=np.bool_)
    filtered_j_i_indicator = np.empty((count, m2), dtype=np.bool_)
    
    # Copy over the rows that pass the threshold.
    idx = 0
    for i in range(n):
        if final_indicator[i]:
            for j in range(m_edges):
                filtered_edges[idx, j] = edges[i, j]
            for j in range(m1):
                filtered_view_from_stack[idx, j] = view_from_stack[i, j]
                filtered_i_j_indicator[idx, j] = i_j_indicator[i, j]
            for j in range(m2):
                filtered_view_to_stack[idx, j] = view_to_stack[i, j]
                filtered_j_i_indicator[idx, j] = j_i_indicator[i, j]
            idx += 1
    
    # Step 4: Compute the broadcasted pairwise AND.
    # For each filtered row, we want to compute a boolean matrix of shape (m1, m2)
    total_elements = count * m1 * m2
    p_lisl_LT_pair = np.empty(total_elements, dtype=np.bool_)
    for i in prange(count):
        for j in range(m1):
            for k in range(m2):
                flat_index = i * (m1 * m2) + j * m2 + k
                p_lisl_LT_pair[flat_index] = filtered_i_j_indicator[i, j] and filtered_j_i_indicator[i, k]
                
    return filtered_edges, filtered_view_from_stack, filtered_view_to_stack, p_lisl_LT_pair

@njit(parallel=True,cache=True)
def compute_directions(positions, edges):
    # positions: array of shape (n, d) with n points in d dimensions
    # edges: array of shape (m, 2) where each row defines an edge by indices into positions
    m = edges.shape[0]
    d = positions.shape[1]
    directions = np.empty((m, d))
    
    # Parallel loop over edges
    for i in prange(m):
        idx0 = edges[i, 0]
        idx1 = edges[i, 1]
        
        # Compute the difference vector
        diff = np.empty(d)
        for j in range(d):
            diff[j] = positions[idx1, j] - positions[idx0, j]
        
        # Compute the Euclidean norm
        norm = 0.0
        for j in range(d):
            norm += diff[j] * diff[j]
        norm = np.sqrt(norm)
        
        # Normalize the difference vector
        for j in range(d):
            directions[i, j] = diff[j] / norm

    return directions

@njit(parallel=True, cache=True)
def compute_view_LT_pair_min_cos(filtered_view_from_stack, filtered_view_to_stack, p_lisl_LT_pair):
    # Assume shapes:
    # filtered_view_from_stack: (A, B)
    # filtered_view_to_stack: (A, C)
    A, B = filtered_view_from_stack.shape
    A2, C = filtered_view_to_stack.shape
    # Allocate an output array for the broadcasted minimum with shape (A, B, C)
    min_vals = np.empty((A, B, C), dtype=filtered_view_from_stack.dtype)
    
    # Manually compute the elementwise minimum for the broadcasted arrays.
    for i in prange(A):
        for j in range(B):
            for k in range(C):
                a_val = filtered_view_from_stack[i, j]
                b_val = filtered_view_to_stack[i, k]
                if a_val < b_val:
                    min_vals[i, j, k] = a_val
                else:
                    min_vals[i, j, k] = b_val
                    
    # Flatten the 3D array to 1D.
    flat_min_vals = min_vals.reshape(-1)
    N = flat_min_vals.shape[0]
    
    # First, count the number of True (or nonzero) entries in the binary indicator.
    count = 0
    for i in range(N):
        if p_lisl_LT_pair[i] != 0:  # Works for both booleans and 0/1 integers.
            count += 1
            
    # Allocate final result with shape (count, 1)
    final_result = np.empty((count, 1), dtype=filtered_view_from_stack.dtype)
    k = 0
    for i in range(N):
        if p_lisl_LT_pair[i] != 0:
            final_result[k, 0] = flat_min_vals[i]
            k += 1
    
    return final_result

@njit(parallel=True, cache=True)
def expand_and_filter_edges(edges, p_lisl_LT_pair, repeat_per_node=4):
    """
    Expand each edge in a grid fashion and filter the expanded arrays based on a boolean mask.
    
    Parameters
    ----------
    edges : np.ndarray
        2D array of shape (num_edges, 2) where each row represents an edge.
    p_lisl_LT_pair : np.ndarray
        Boolean 1D array of length (num_edges * repeat_per_node^2) indicating which expanded entries to keep.
    repeat_per_node : int, optional
        Number of repetitions per node (default is 4). The expansion grid will have shape (repeat_per_node, repeat_per_node).
    
    Returns
    -------
    filtered_repeated : np.ndarray
        Filtered array of repeated original edges (shape (num_selected, 2)).
    filtered_expanded : np.ndarray
        Filtered array of expanded edges with grid offsets (shape (num_selected, 2)).
    """
    num_edges = edges.shape[0]
    num_grid = repeat_per_node * repeat_per_node
    total = num_edges * num_grid

    # Step 1: Compute prefix sum over the boolean mask (serial loop).
    cumsum = np.empty(total, dtype=np.int64)
    count = 0
    for i in range(total):
        if p_lisl_LT_pair[i]:
            count += 1
        cumsum[i] = count
    total_true = count

    # Allocate output arrays.
    filtered_repeated = np.empty((total_true, 2), dtype=edges.dtype)
    filtered_expanded = np.empty((total_true, 2), dtype=edges.dtype)

    # Step 2: Loop over all expansion entries in parallel.
    # For each flattened index, if the mask is True, compute the corresponding edge expansion and copy it.
    for i in prange(total):
        if p_lisl_LT_pair[i]:
            # Determine output position from prefix sum.
            pos = cumsum[i] - 1  # Adjust for 0-indexing.
            # Map flattened index to original edge index and grid index.
            e = i // num_grid
            g = i % num_grid
            # Original (repeated) edge.
            filtered_repeated[pos, 0] = edges[e, 0]
            filtered_repeated[pos, 1] = edges[e, 1]
            # Compute grid offsets.
            offset0 = g // repeat_per_node  # row offset
            offset1 = g % repeat_per_node   # column offset
            # Expanded edge: multiply the original edge by repeat_per_node and add the grid offset.
            filtered_expanded[pos, 0] = edges[e, 0] * repeat_per_node + offset0
            filtered_expanded[pos, 1] = edges[e, 1] * repeat_per_node + offset1

    return filtered_repeated, filtered_expanded

@njit(parallel=True,cache=True)
def compute_weighted_edges(velocities, positions, filtered_repeated, 
                           filtered_expanded, view_LT_pair_min_cos, FOR_THETA, MIN_TIME=100):
    n = filtered_repeated.shape[0]
    
    # Preallocate arrays for intermediate computations.
    relative_speed = np.empty((n, 3), dtype=velocities.dtype)
    relative_direction = np.empty((n, 3), dtype=positions.dtype)
    cross = np.empty((n, 3), dtype=velocities.dtype)
    angular_speed = np.empty(n, dtype=velocities.dtype)
    view_time_approx = np.empty(n, dtype=velocities.dtype)
    
    theta_rad = math.radians(FOR_THETA)
    
    # Compute relative values in parallel.
    for i in prange(n):
        idx0 = filtered_repeated[i, 0]
        idx1 = filtered_repeated[i, 1]
        
        # Calculate relative speed and direction for each axis.
        for j in range(3):
            relative_speed[i, j] = velocities[idx1, j] - velocities[idx0, j]
            relative_direction[i, j] = positions[idx1, j] - positions[idx0, j]
        
        # Manually compute the cross product.
        cross[i, 0] = relative_speed[i, 1]*relative_direction[i, 2] - relative_speed[i, 2]*relative_direction[i, 1]
        cross[i, 1] = relative_speed[i, 2]*relative_direction[i, 0] - relative_speed[i, 0]*relative_direction[i, 2]
        cross[i, 2] = relative_speed[i, 0]*relative_direction[i, 1] - relative_speed[i, 1]*relative_direction[i, 0]
        
        # Compute norms.
        cross_norm = math.sqrt(cross[i, 0]**2 + cross[i, 1]**2 + cross[i, 2]**2)
        dir_norm = math.sqrt(relative_direction[i, 0]**2 + relative_direction[i, 1]**2 + relative_direction[i, 2]**2)
        
        # Avoid division by zero.
        if dir_norm == 0:
            angular_speed[i] = 0.0
        else:
            angular_speed[i] = cross_norm / dir_norm
        
        # Compute view time approximation.
        # Note: view_LT_pair_min_cos[i] should be in the domain of acos, i.e. between -1 and 1.
        acos_val = math.acos(view_LT_pair_min_cos[i])
        # Protect against angular_speed being zero.
        if angular_speed[i] == 0:
            view_time_approx[i] = 1e10  # Use a large number to represent near-infinite view time.
        else:
            view_time_approx[i] = (theta_rad - acos_val) / math.fabs(angular_speed[i])
    
    # Count how many edges meet the criterion.
    count = 0
    for i in range(n):
        if view_time_approx[i] > MIN_TIME:
            count += 1

    # Determine number of columns in filtered_expanded.
    num_cols = filtered_expanded.shape[1]
    # Allocate result array: each row is filtered_expanded row concatenated with one view_time value.
    res = np.empty((count, num_cols + 1), dtype=filtered_expanded.dtype)
    
    k = 0
    for i in range(n):
        if view_time_approx[i] > 100:
            # Copy the corresponding row from filtered_expanded.
            for j in range(num_cols):
                res[k, j] = filtered_expanded[i, j]
            # Append the view time value.
            res[k, num_cols] = view_time_approx[i]
            k += 1
    
    return res

@njit(parallel=True,cache=True)
def compute_view_stacks(lct_directions, edges, direction):
    num_edges = edges.shape[0]
    # Pre-allocate output arrays.
    view_from_stack = np.empty((num_edges, 4), dtype=direction.dtype)
    view_to_stack = np.empty((num_edges, 4), dtype=direction.dtype)
    
    for i in prange(num_edges):
        # Get the source and destination indices for this edge.
        src = edges[i, 0]
        dst = edges[i, 1]
        d0 = direction[i, 0]
        d1 = direction[i, 1]
        d2 = direction[i, 2]
        
        # Compute dot products for source side.
        view_from_stack[i, 0] = lct_directions[src, 0, 0]*d0 + lct_directions[src, 0, 1]*d1 + lct_directions[src, 0, 2]*d2
        view_from_stack[i, 1] = lct_directions[src, 1, 0]*d0 + lct_directions[src, 1, 1]*d1 + lct_directions[src, 1, 2]*d2
        view_from_stack[i, 2] = lct_directions[src, 2, 0]*d0 + lct_directions[src, 2, 1]*d1 + lct_directions[src, 2, 2]*d2
        view_from_stack[i, 3] = lct_directions[src, 3, 0]*d0 + lct_directions[src, 3, 1]*d1 + lct_directions[src, 3, 2]*d2
        
        # Compute dot products for destination side with -direction.
        view_to_stack[i, 0] = lct_directions[dst, 0, 0]*(-d0) + lct_directions[dst, 0, 1]*(-d1) + lct_directions[dst, 0, 2]*(-d2)
        view_to_stack[i, 1] = lct_directions[dst, 1, 0]*(-d0) + lct_directions[dst, 1, 1]*(-d1) + lct_directions[dst, 1, 2]*(-d2)
        view_to_stack[i, 2] = lct_directions[dst, 2, 0]*(-d0) + lct_directions[dst, 2, 1]*(-d1) + lct_directions[dst, 2, 2]*(-d2)
        view_to_stack[i, 3] = lct_directions[dst, 3, 0]*(-d0) + lct_directions[dst, 3, 1]*(-d1) + lct_directions[dst, 3, 2]*(-d2)

    return view_from_stack, view_to_stack

@njit(parallel=True,cache=True)
def apply_mask_to_view(mask, edges, view_from_stack, view_to_stack):
    num_edges = edges.shape[0]
    # Pre-allocate output arrays.
    ret_view_from_stack = np.empty_like(view_from_stack)
    ret_view_to_stack = np.empty_like(view_to_stack)
    
    for i in prange(num_edges):
        # Get the source and destination indices for this edge.
        src = edges[i, 0]
        dst = edges[i, 1]
        
        # Compute dot products for source side.
        ret_view_from_stack[i, 0] = mask[src, 0] * view_from_stack[i, 0]
        ret_view_from_stack[i, 1] = mask[src, 1] * view_from_stack[i, 1]
        ret_view_from_stack[i, 2] = mask[src, 2] * view_from_stack[i, 2]
        ret_view_from_stack[i, 3] = mask[src, 3] * view_from_stack[i, 3]
        
        ret_view_to_stack[i, 0] = mask[dst, 0] * view_to_stack[i, 0]
        ret_view_to_stack[i, 1] = mask[dst, 1] * view_to_stack[i, 1]
        ret_view_to_stack[i, 2] = mask[dst, 2] * view_to_stack[i, 2]
        ret_view_to_stack[i, 3] = mask[dst, 3] * view_to_stack[i, 3]

    return ret_view_from_stack, ret_view_to_stack

@njit(parallel=True,cache=True)
def optimize_edge_and_color_data(edges_color, connected_sat, connected_lct, positions, lift=False):
    M = connected_lct.shape[0]
    M_sat = connected_sat.shape[0]
    
    # Build edges_color_from and edges_color_to.
    # Each is computed by taking the row from edges_color indexed by connected_edges mod 4,
    # duplicating it, then reshaping to yield an array of shape (2*M, 4).
    edges_color_from = np.empty((2 * M, 4), dtype=edges_color.dtype)
    edges_color_to   = np.empty((2 * M, 4), dtype=edges_color.dtype)
    
    for i in prange(M):
        idx_from = connected_lct[i, 0] % 4
        idx_to   = connected_lct[i, 1] % 4
        # Duplicate the row for "from" and "to".
        for j in range(4):
            edges_color_from[2 * i, j]     = edges_color[idx_from, j]
            edges_color_from[2 * i + 1, j] = edges_color[idx_from, j]
            edges_color_to[2 * i, j]       = edges_color[idx_to, j]
            edges_color_to[2 * i + 1, j]   = edges_color[idx_to, j]
    
    # Concatenate edges_color_from and edges_color_to along axis 0.
    edges_color_data = np.empty((4 * M, 4), dtype=edges_color.dtype)
    for i in prange(2 * M):
        for j in range(4):
            edges_color_data[i, j] = edges_color_from[i, j]
            edges_color_data[i + 2 * M, j] = edges_color_to[i, j]

    # Compute positions for connected_sat.
    p_from = np.empty((M_sat, 3), dtype=positions.dtype)
    p_to   = np.empty((M_sat, 3), dtype=positions.dtype)
    p_mid  = np.empty((M_sat, 3), dtype=positions.dtype)
    
    for i in prange(M_sat):
        idx_from = connected_sat[i, 0]
        idx_to   = connected_sat[i, 1]
        for j in range(3):
            # Multiply by a slight factor (1.0001)
            if lift:
                p_from[i, j] = positions[idx_from, j] * 1.0001
                p_to[i, j]   = positions[idx_to, j] * 1.0001
            else:
                p_from[i, j] = positions[idx_from, j]
                p_to[i, j]   = positions[idx_to, j]
            # p_mid is computed as the average.
            p_mid[i, j]  = (p_from[i, j] + p_to[i, j]) * 0.5

    # Build c_lisl_data_from = reshape(concatenate(p_from, p_mid, axis=1)) => shape (2*M_sat, 3)
    c_lisl_data_from = np.empty((2 * M_sat, 3), dtype=positions.dtype)
    for i in prange(M_sat):
        for j in range(3):
            c_lisl_data_from[2 * i, j]     = p_from[i, j]
            c_lisl_data_from[2 * i + 1, j] = p_mid[i, j]
    
    # Build c_lisl_data_to = reshape(concatenate(p_mid, p_to, axis=1)) => shape (2*M_sat, 3)
    c_lisl_data_to = np.empty((2 * M_sat, 3), dtype=positions.dtype)
    for i in prange(M_sat):
        for j in range(3):
            c_lisl_data_to[2 * i, j]     = p_mid[i, j]
            c_lisl_data_to[2 * i + 1, j] = p_to[i, j]
    
    # Concatenate c_lisl_data_from and c_lisl_data_to along axis 0.
    c_lisl_data = np.empty((4 * M_sat, 3), dtype=positions.dtype)
    for i in prange(2 * M_sat):
        for j in range(3):
            c_lisl_data[i, j] = c_lisl_data_from[i, j]
            c_lisl_data[i + 2 * M_sat, j] = c_lisl_data_to[i, j]
    
    return edges_color_data, c_lisl_data