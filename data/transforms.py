import torch
from torch_geometric.data import Data


def build_spatial_edges(H: int, W: int, periodic_lon: bool = True, neighborhood: int = 4):
    """
    Build spatial adjacency edges for a 2D HxW grid.

    Each grid cell is treated as a graph node. Nodes are connected to their
    spatial neighbors depending on the neighborhood size:

    - 4-neighborhood: connects North, South, East, West
    - 8-neighborhood: includes diagonals (NW, NE, SW, SE) in addition to the 4-neighborhood

    Args:
        H (int): Grid height (# of latitude points)
        W (int): Grid width (# of longitude points)
        periodic_lon (bool): If True, wrap around at the longitude edges (periodic boundary in W)
        neighborhood (int): Either 4 or 8, determining the connectivity

    Returns:
        torch.Tensor: Edge indices with shape [2, num_edges]
    """
    edges = []
    for i in range(H):
        for j in range(W):
            node = i * W + j

            # --- 4-neighborhood ---
            if i > 0:  # North
                edges.append([node, (i - 1) * W + j])
            if i < H - 1:  # South
                edges.append([node, (i + 1) * W + j])

            if j > 0:  # West
                edges.append([node, i * W + (j - 1)])
            elif periodic_lon:  # Wrap around west edge to east edge
                edges.append([node, i * W + (W - 1)])

            if j < W - 1:  # East
                edges.append([node, i * W + (j + 1)])
            elif periodic_lon:  # Wrap around east edge to west edge
                edges.append([node, i * W + 0])

            # --- Extra diagonals for 8-neighborhood ---
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

def build_spatio_temporal_edges(H: int, W: int, T: int,
                                periodic_lon: bool = True,
                                neighborhood: int = 4,
                                graph_type: str = "cartesian"):
    """
    Efficient, vectorized construction of spatio-temporal edges.

    - Spatial edges: within each timestep.
    - Temporal edges:
        * "cartesian": node_t → node_{t+1}
        * "strong": node_t → itself_{t+1} and neighbors_{t+1}
    """
    spatial_edges = build_spatial_edges(H, W, periodic_lon, neighborhood)
    num_nodes_per_t = H * W

    # Spatial edges for all timesteps 
    spatial_offsets = torch.arange(T) * num_nodes_per_t
    spatial_edges_all = (
        spatial_edges.unsqueeze(0) + spatial_offsets.view(-1, 1, 1)
    ).reshape(-1, 2).t()  # shape [2, T * num_spatial_edges]

    # Temporal edges 
    if T > 1:
        # Offsets between consecutive timesteps
        t_offsets = torch.arange(T - 1) * num_nodes_per_t
        next_offsets = t_offsets + num_nodes_per_t

        # Cartesian temporal edges (self to self) 
        self_edges = torch.arange(num_nodes_per_t)
        self_edges = torch.stack([
            self_edges.repeat(T - 1) + t_offsets.repeat_interleave(num_nodes_per_t),
            self_edges.repeat(T - 1) + next_offsets.repeat_interleave(num_nodes_per_t)
        ])

        if graph_type == "cartesian":
            temporal_edges_all = self_edges

        elif graph_type == "strong":
            # Neighbor-to-next-timestep edges 
            edges_next = spatial_edges + num_nodes_per_t  # shift to next timestep indices
            num_spatial_edges = edges_next.shape[1]

            # Repeat spatial pattern for all T-1 transitions
            t_offsets = torch.arange(T - 1, device=edges_next.device) * num_nodes_per_t

            src_offsets = t_offsets.repeat_interleave(num_spatial_edges)
            dst_offsets = t_offsets.repeat_interleave(num_spatial_edges)

            strong_edges = torch.vstack([
                (edges_next[0].repeat(T - 1) + src_offsets),
                (edges_next[1].repeat(T - 1) + dst_offsets)
            ])

            # Combine self and neighbor links
            temporal_edges_all = torch.cat([self_edges, strong_edges], dim=1)
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
    else:
        temporal_edges_all = torch.empty((2, 0), dtype=torch.long)

    # Combine all
    return torch.cat([spatial_edges_all, temporal_edges_all], dim=1).contiguous()
  


def to_spatio_temporal_graph(X: torch.Tensor, y: torch.Tensor,
                             H: int, W: int,
                             neighborhood: int = 4,
                             graph_type: str = "cartesian"):
    """
    Convert an ERA5 sample into a spatio-temporal PyG Data object.

    Each grid cell at each timestep becomes a node. 
    Spatial and temporal edges are constructed according to the chosen settings.

    Args:
        X (Tensor): Input sequence of shape (T, C, H, W)
        y (Tensor): Target at forecast horizon, shape (C, H, W)
        H (int): Grid height
        W (int): Grid width
        neighborhood (int): 4 or 8 spatial neighbors
        graph_type (str): "cartesian" (self temporal links) or 
                          "strong" (spatial neighbors across timesteps)

    Returns:
        Data: torch_geometric.data.Data object with:
              - x: Node features, shape [T*H*W, C]
              - edge_index: Graph edges, shape [2, num_edges]
              - y: Node labels, shape [H*W, C]
    """
    T, C, _, _ = X.shape
    num_nodes = H * W * T

    # Node features: flatten (T, H, W, C) to (T*H*W, C)
    X_nodes = X.permute(0, 2, 3, 1).reshape(num_nodes, C)

    # Labels: flatten target (C, H, W) to (H*W, C)
    C_y = y.shape[0]
    y_nodes = y.permute(1, 2, 0).reshape(H * W, C_y)

    # Build edges
    edge_index = build_spatio_temporal_edges(
        H, W, T, neighborhood=neighborhood, graph_type=graph_type
    )

    return Data(x=X_nodes, edge_index=edge_index, y=y_nodes)
