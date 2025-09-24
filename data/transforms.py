import torch
from torch_geometric.data import Data


def build_spatial_edges(H: int, W: int, periodic_lon: bool = True):
    """Build spatial adjacency for HxW grid."""
    edges = []
    for i in range(H):
        for j in range(W):
            node = i * W + j
            # North
            if i > 0:
                edges.append([node, (i - 1) * W + j])
            # South
            if i < H - 1:
                edges.append([node, (i + 1) * W + j])
            # West
            if j > 0:
                edges.append([node, i * W + (j - 1)])
            elif periodic_lon:
                edges.append([node, i * W + (W - 1)])
            # East
            if j < W - 1:
                edges.append([node, i * W + (j + 1)])
            elif periodic_lon:
                edges.append([node, i * W + 0])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, num_edges]


def build_spatio_temporal_edges(H: int, W: int, T: int, periodic_lon: bool = True):
    """
    Build edges for spatio-temporal graph.
    - Spatial edges within each timestep.
    - Temporal edges across timesteps.
    """
    spatial_edges = build_spatial_edges(H, W, periodic_lon)
    all_edges = []

    for t in range(T):
        offset = t * H * W
        # Add spatial edges at this timestep
        edges_t = spatial_edges + offset
        all_edges.append(edges_t)

        # Add temporal edges (self-links from t → t+1)
        if t < T - 1:
            for i in range(H * W):
                all_edges.append(
                    torch.tensor([[offset + i], [offset + i + H * W]], dtype=torch.long)
                )

    return torch.cat(all_edges, dim=1)  # [2, total_edges]


def to_spatio_temporal_graph(X: torch.Tensor, y: torch.Tensor, H: int, W: int):
    """
    Convert ERA5 sample to spatio-temporal PyG graph.
    
    Args:
        X (Tensor): (T, C, H, W) input sequence
        y (Tensor): (C, H, W) target at forecast horizon
        H, W (int): grid size
    Returns:
        Data: PyG Data object
    """
    T, C, _, _ = X.shape
    num_nodes = H * W * T

    # Node features: flatten grid per timestep
    X_nodes = X.permute(0, 2, 3, 1).reshape(num_nodes, C)  # [T*H*W, C]

    # Labels: flatten target
    y_nodes = y.permute(1, 2, 0).reshape(H * W, C)  # [H*W, C]

    # Build edges
    edge_index = build_spatio_temporal_edges(H, W, T)

    return Data(x=X_nodes, edge_index=edge_index, y=y_nodes)
