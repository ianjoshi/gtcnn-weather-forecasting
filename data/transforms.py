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
    Build edges for a spatio-temporal graph over an HxW grid sequence of length T.

    - Always includes spatial edges within each timestep (from build_spatial_edges).
    - Temporal edges differ depending on `graph_type`:

      * cartesian: 
          Each node at time t connects only to itself at time t+1 
          (temporal self-links).
      * strong:
          Each node at time t connects to its spatial neighbors at time t+1 
          (spatio-temporal neighbor propagation).

    Args:
        H (int): Grid height
        W (int): Grid width
        T (int): Sequence length (timesteps)
        periodic_lon (bool): If True, wrap around longitude boundaries
        neighborhood (int): 4 or 8 spatial neighbors
        graph_type (str): "cartesian" or "strong"

    Returns:
        torch.Tensor: Edge indices with shape [2, total_edges]
    """
    spatial_edges = build_spatial_edges(H, W, periodic_lon, neighborhood)
    all_edges = []

    for t in range(T):
        offset = t * H * W
        # --- Spatial edges at timestep t ---
        edges_t = spatial_edges + offset
        all_edges.append(edges_t)

        if t < T - 1:
            next_offset = (t + 1) * H * W

            if graph_type == "cartesian":
                # Temporal edges: self-links only (node_t to node_{t+1})
                for i in range(H * W):
                    all_edges.append(
                        torch.tensor([[offset + i], [next_offset + i]], dtype=torch.long)
                    )

            elif graph_type == "strong":
                # Temporal edges: connect to neighbors at next timestep
                edges_next = spatial_edges + next_offset
                strong_edges = torch.vstack([
                    edges_next[0] - H * W,  # shift sources back one timestep
                    edges_next[1]
                ])
                all_edges.append(strong_edges)

            else:
                raise ValueError(f"Unknown graph_type: {graph_type}")

    return torch.cat(all_edges, dim=1)  


def to_spatio_temporal_graph(X: torch.Tensor, y: torch.Tensor,
                             H: int, W: int,
                             neighborhood: int = 4,
                             graph_type: str = "cartesian"):
    """
    Convert an ERA5 sample into a spatio-temporal PyG Data object.

    Args:
        X: (T, Cx, H, W)   # Cx may include extra channels (e.g., sin/cos temporal encodings)
        y: (Cy, H, W)      # Cy = number of physical variables only
    Returns:
        Data with:
          - x: [T*H*W, Cx]
          - y: [H*W,   Cy]
    """
    T, Cx, _, _ = X.shape            # channels from X (e.g., vars + 2 temporal)
    Cy = y.shape[0]                  # channels from y (only physical vars)

    # assert that y doesn't accidentally include the temporal channels
    assert Cy <= Cx, f"y has more channels ({Cy}) than X ({Cx}); check preprocessing."

    num_nodes = H * W * T

    # Node features from X: (T, H, W, Cx) -> (T*H*W, Cx)
    X_nodes = X.permute(0, 2, 3, 1).reshape(num_nodes, Cx)

    # Labels from y: (H, W, Cy) -> (H*W, Cy)
    y_nodes = y.permute(1, 2, 0).reshape(H * W, Cy)

    # Build edges
    edge_index = build_spatio_temporal_edges(
        H, W, T, neighborhood=neighborhood, graph_type=graph_type
    )

    return Data(x=X_nodes, edge_index=edge_index, y=y_nodes)

