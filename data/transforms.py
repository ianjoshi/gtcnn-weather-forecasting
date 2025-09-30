import torch
from torch_geometric.data import Data


def _build_cartesian_edges(H: int, W: int, periodic_lon: bool = True, neighborhood: int = 4):
    """Build spatial adjacency for HxW grid with 4- or 8-neighborhood."""
    edges = []
    for i in range(H):
        for j in range(W):
            node = i * W + j

            # 4-neighbors (N, S, W, E)
            if i > 0:
                edges.append([node, (i - 1) * W + j])   # North

            if i < H - 1:
                edges.append([node, (i + 1) * W + j])   # South

            if j > 0:
                edges.append([node, i * W + (j - 1)])   # West
            elif periodic_lon:
                edges.append([node, i * W + (W - 1)])
                
            if j < W - 1:
                edges.append([node, i * W + (j + 1)])   # East
            elif periodic_lon:
                edges.append([node, i * W + 0])

            # Diagonals if neighborhood=8
            if neighborhood == 8:
                # North-west
                if i > 0 and (j > 0 or periodic_lon):
                    jj = j - 1 if j > 0 else W - 1
                    edges.append([node, (i - 1) * W + jj])
                # North-east
                if i > 0 and (j < W - 1 or periodic_lon):
                    jj = j + 1 if j < W - 1 else 0
                    edges.append([node, (i - 1) * W + jj])
                # South-west
                if i < H - 1 and (j > 0 or periodic_lon):
                    jj = j - 1 if j > 0 else W - 1
                    edges.append([node, (i + 1) * W + jj])
                # South-east
                if i < H - 1 and (j < W - 1 or periodic_lon):
                    jj = j + 1 if j < W - 1 else 0
                    edges.append([node, (i + 1) * W + jj])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_spatial_edges(H: int, W: int, periodic_lon: bool = True,
                        neighborhood: int = 4):
    """Build spatial adjacency for HxW grid (cartesian only)."""
    return _build_cartesian_edges(H, W, periodic_lon, neighborhood)


def build_spatio_temporal_edges(H: int, W: int, T: int,
                                periodic_lon: bool = True,
                                neighborhood: int = 4,
                                graph_type: str = "cartesian"):
    """
    Build edges for spatio-temporal graph.
    - Spatial edges within each timestep.
    - Temporal edges across timesteps:
        * cartesian: self-links only
        * strong: links to spatial neighbors in next timestep
    """
    spatial_edges = build_spatial_edges(H, W, periodic_lon, neighborhood)
    all_edges = []

    for t in range(T):
        offset = t * H * W
        # Spatial edges at timestep t
        edges_t = spatial_edges + offset
        all_edges.append(edges_t)

        if t < T - 1:
            next_offset = (t + 1) * H * W

            if graph_type == "cartesian":
                # Temporal self-links (node -> same node at t+1)
                for i in range(H * W):
                    all_edges.append(
                        torch.tensor([[offset + i], [next_offset + i]], dtype=torch.long)
                    )

            elif graph_type == "strong":
                # Strong temporal: link node to its neighbors at next timestep
                # reuse spatial_edges but shift targets to next timestep
                edges_next = spatial_edges + next_offset
                strong_edges = torch.vstack([
                    edges_next[0] - H * W,  # shift sources back one timestep
                    edges_next[1]
                ])
                all_edges.append(strong_edges)

            else:
                raise ValueError(f"Unknown graph_type: {graph_type}")

    return torch.cat(all_edges, dim=1)  # [2, total_edges]


def to_spatio_temporal_graph(X: torch.Tensor, y: torch.Tensor,
                             H: int, W: int,
                             neighborhood: int = 4,
                             graph_type: str = "cartesian"):
    """
    Convert ERA5 sample to spatio-temporal PyG graph.
    
    Args:
        X (Tensor): (T, C, H, W) input sequence
        y (Tensor): (C, H, W) target at forecast horizon
        H, W (int): grid size
        neighborhood (int): 4 or 8 neighbors
        graph_type (str): "cartesian" or "strong"
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
    edge_index = build_spatio_temporal_edges(
        H, W, T, neighborhood=neighborhood, graph_type=graph_type
    )

    return Data(x=X_nodes, edge_index=edge_index, y=y_nodes)
