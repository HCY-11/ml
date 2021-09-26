import mnist
import numpy as np
from numba import njit, jit, int64

@njit
def unravel_index(idx, num_cols):
    row = idx // num_cols
    col = idx - row * num_cols

    return row, col

@njit
def relu(x):
    return np.maximum(0, x)

@njit
def relu_prime(y):
    return y > 0

@njit(fastmath=True)
def softmax(x):
    out = np.zeros_like(x)

    size = len(x)
    total = np.sum(np.exp(x), axis=0)

    for i in range(size):
        out[i] = np.exp(x[i]) / total
    
    return out

@njit
def cross_entropy_loss(p_hat_i):
    return -np.log(p_hat_i)

@njit(fastmath=True)
def conv_jit(Xi, filters, conv_b):
    num_filters = filters.shape[0]
    filter_size = filters.shape[1]

    feature_map_size = Xi.shape[0] - filters.shape[1] + 1
    feature_maps = np.zeros((num_filters, feature_map_size, feature_map_size))

    for i in range(num_filters):
        for j in range(feature_map_size):
            for k in range(feature_map_size):
                row_stop = j + filter_size
                col_stop = k + filter_size
                feature_maps[i][j][k] = np.sum(filters[i] * Xi[j:row_stop, k:col_stop]) + conv_b[i]
    
    return feature_maps

@njit(fastmath=True)
def conv_back_jit(Xi, max_pool_back, feature_maps, filter_shape):
    num_filters = filter_shape[0]
    filter_size = filter_shape[1]

    b_adj = np.zeros(num_filters)
    conv_adj = np.zeros(filter_shape)

    for i in range(num_filters):
        for j in range(feature_maps.shape[1]):
            for k in range(feature_maps.shape[1]):
                row_stop = j + filter_size
                col_stop = k + filter_size

                delta = max_pool_back[i][j][k] * relu_prime(feature_maps[i][j][k])
                b_adj[i] += delta
                conv_adj[i] += delta * Xi[j:row_stop, k:col_stop]
        
    return conv_adj, b_adj

@njit(fastmath=True)
def max_pool_jit(feature_maps, num_filters, pool_step=2, pool_section=2):
    feature_map_size = feature_maps.shape[1]
    pool_size = feature_map_size // 2

    max_pools = np.zeros((num_filters, pool_size, pool_size))
    max_indices = np.zeros((num_filters, pool_size, pool_size, 2), dtype=int64)

    for i in range(num_filters):
        for j in range(0, feature_map_size, pool_step):
            for k in range(0, feature_map_size, pool_step):
                # Max pooling using 2x2 sections
                row_stop = j + pool_section
                col_stop = k + pool_section
                pool_row = int(j / 2)
                pool_col = int(k / 2)

                image_chunk = feature_maps[i][j:row_stop, k:col_stop]
                max_val = np.max(image_chunk) # Get max value within 2x2 section

                x, y = unravel_index(np.argmax(image_chunk), image_chunk.shape[1])

                max_indices[i][pool_row][pool_col] = (j + x, k + y)
                max_pools[i][pool_row][pool_col] = max_val

    return max_pools, max_indices

@njit(fastmath=True)
def max_pool_back_jit(fc_err, fc_w, max_indices, feature_maps_shape, num_filters, pool_step=2, pool_section=2):
    err = fc_w.T @ fc_err 
    err = err.reshape(max_indices.shape[:3])

    out = np.zeros(feature_maps_shape)

    for i in range(num_filters):
        for j in range(max_indices.shape[1]):
            for k in range(max_indices.shape[1]):
                x, y = max_indices[i][j][k]
                out[i][x][y] = err[i][j][k] 

    return out

@njit(fastmath=True)
def fully_connected_jit(max_pools, w, b):
    x = max_pools.flatten()

    out_size = len(w)
    raw_out = np.zeros(out_size)

    for i in range(out_size):
        raw_out[i] = np.dot(w[i], x) + b[i]

    out = softmax(raw_out)
    return out

@jit(fastmath=True)
def fully_connected_back_jit(fc_out, yi, max_pools):
    loss = cross_entropy_loss(fc_out[yi])

    err = fc_out
    err[yi] -= 1

    fc_in = max_pools.flatten()
    b_adj = err
    w_adj = np.dot(b_adj.reshape(-1, 1), fc_in.reshape(-1, 1).T)

    return w_adj, b_adj, loss, err