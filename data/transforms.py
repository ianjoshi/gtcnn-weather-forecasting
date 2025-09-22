import torch
import numpy as np

def normalize_tensor(x, mean=None, std=None):
    """
    Normalize tensor with optional provided mean/std.
    """
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    return (x - mean) / (std + 1e-8), mean, std


def grid_to_graph(lat_dim, lon_dim, connectivity=4):
    """
    Build adjacency list for a regular lat-lon grid.
    
    Args:
        lat_dim (int): number of latitude points
        lon_dim (int): number of longitude points
        connectivity (int): 4 (N,S,E,W) or 8 (include diagonals)
    Returns:
        edge_index (torch.LongTensor): [2, num_edges]
    """
    edges = []
    for i in range(lat_dim):

        for j in range(lon_dim):

            idx = i * lon_dim + j
            neighbors = []

            # NORTH
            if i > 0: 
                neighbors.append(((i-1)*lon_dim + j))  

            # SOUTH
            if i < lat_dim-1: 
                neighbors.append(((i+1)*lon_dim + j)) 

            # WEST
            if j > 0: 
                neighbors.append((i*lon_dim + j-1))  
                         
            # EAST
            if j < lon_dim-1: 
                neighbors.append((i*lon_dim + j+1))  
            
            # DIAGONALS
            if connectivity == 8:  
                if i > 0 and j > 0: 
                    neighbors.append((i-1)*lon_dim + (j-1))
                if i > 0 and j < lon_dim-1: 
                    neighbors.append((i-1)*lon_dim + (j+1))
                if i < lat_dim-1 and j > 0: 
                    neighbors.append((i+1)*lon_dim + (j-1))
                if i < lat_dim-1 and j < lon_dim-1: 
                    neighbors.append((i+1)*lon_dim + (j+1))
            
            for n in neighbors:
                edges.append([idx, n])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()
