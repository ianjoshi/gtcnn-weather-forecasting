from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GraphNorm

class SIGN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 3,
        dropout: float = 0.1,
        use_bn: bool = True,
    ) -> None:
        """
        SIGN: Scalable Inception Graph Neural Network

        Args:
            in_channels: input node feature dim
            hidden_channels: hidden feature dim
            out_channels: output feature dim
            K: number of hops (A^k)
            dropout: dropout prob
            use_bn: use GraphNorm between layers
        """
        super().__init__()

        self.K = K
        self.use_bn = use_bn
        self.dropout = dropout
        
        # Store precomputed adjacency matrix powers as PyTorch sparse tensors
        self.precomputed_adj: Optional[Dict[int, torch.Tensor]] = None

        # Normalize after Θk transformations (Option B-2)
        # This follows standard deep learning practice: Transform → Normalize → Activate
        if use_bn:
            self.theta_norms = nn.ModuleList()
            for k in range(K + 1):  # K+1 norms for all transformed features
                self.theta_norms.append(GraphNorm(hidden_channels))
        else:
            self.theta_norms = None
        
        # SIGN architecture: separate learnable matrix for each shift
        # Θ0, Θ1, ..., Θr in the paper
        self.theta_layers = nn.ModuleList()
        for k in range(K + 1):  # K+1 because we include X (no shift)
            self.theta_layers.append(nn.Linear(in_channels, hidden_channels))
        
        # Final combination layer Ω in the paper
        self.omega = nn.Linear(hidden_channels * (K + 1), out_channels)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def precompute_adjacency_powers(self, edge_index, N, T):
        """
        Precompute adjacency matrix powers [A, A^2, ..., A^K] for a SINGLE graph.
        This is the actual SIGN precomputation - store the matrices, not the features!
        
        Args:
            edge_index: [2, E] - spatio-temporal edge indices for ONE sample
            N: number of spatial nodes (H*W)
            T: number of timesteps
        """
        num_nodes = N * T
        print(f"Precomputing adjacency matrix powers for K={self.K}...")
        print(f"  Single graph structure: {N} spatial nodes × {T} timesteps = {num_nodes} total nodes")
        
        # Add self-loops
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Compute D^(-1/2) for symmetric normalization
        row, col = edge_index_with_loops
        deg = degree(row, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Create normalized adjacency: D^(-1/2) @ A @ D^(-1/2)
        values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj_norm = torch.sparse_coo_tensor(
            edge_index_with_loops,
            values,
            (num_nodes, num_nodes)
        ).coalesce()

        # Precompute powers: {1: A, 2: A^2, 3: A^3, ...}
        self.precomputed_adj = {}
        current_adj = adj_norm
        
        for k in range(1, self.K + 1):
            if k == 1:
                self.precomputed_adj[k] = current_adj
            else:
                # Sparse @ Sparse matrix multiplication
                # IMPORTANT: coalesce() combines duplicate indices for faster operations
                current_adj = torch.sparse.mm(current_adj, adj_norm).coalesce()
                self.precomputed_adj[k] = current_adj
        
        print(f"Precomputation complete. Stored {len(self.precomputed_adj)} adjacency powers.")

    def forward(self, x, edge_index, N, T):
        """
        Forward pass supporting batched graphs with identical structure.
        
        Args:
            x: [B*N*T, C_in] - features for B graphs stacked
            edge_index: [2, E] - edges (only used for consistency, not needed)
            N: spatial nodes per time slice (H*W)
            T: number of time steps in input sequence
            
        Returns:
            y_hat: [B*N, C_out] - predictions for last timestep of each graph
        """
        nodes_per_graph = N * T
        batch_size = x.size(0) // nodes_per_graph
        
        # Reshape to separate batch dimension: [B*N*T, C] -> [B, N*T, C]
        x_batched = x.view(batch_size, nodes_per_graph, -1)
        
        # Apply SIGN transformations to all graphs, then normalize across batch
        all_transformed = []  # Store [K+1] tensors of shape [B, N*T, hidden]
        
        for k in range(self.K + 1):
            batch_features = []
            for b in range(batch_size):
                x_single = x_batched[b]  # [N*T, C_in]
                
                # Apply shift (or use original for k=0)
                if k == 0:
                    x_k = x_single
                else:
                    x_k = torch.sparse.mm(self.precomputed_adj[k], x_single)
                
                # Apply learnable transformation Θk
                x_theta_k = self.theta_layers[k](x_k)  # [N*T, hidden]
                batch_features.append(x_theta_k)
            
            # Stack batch: [B, N*T, hidden]
            x_theta_batch = torch.stack(batch_features, dim=0)
            
            # Normalize across ENTIRE batch (like GTCNN)
            if self.theta_norms is not None:
                # Reshape to [B*N*T, hidden] for GraphNorm
                x_theta_flat = x_theta_batch.view(batch_size * nodes_per_graph, -1)
                x_theta_flat = self.theta_norms[k](x_theta_flat)
                x_theta_batch = x_theta_flat.view(batch_size, nodes_per_graph, -1)
            
            all_transformed.append(x_theta_batch)
        
        # Concatenate along feature dimension: [B, N*T, hidden*(K+1)]
        z_batch = torch.cat(all_transformed, dim=-1)
        
        # Apply non-linearity and dropout
        z_batch = F.relu(z_batch)
        z_batch = self.dropout_layer(z_batch)
        
        # Final transformation
        z_flat = z_batch.view(batch_size * nodes_per_graph, -1)  # [B*N*T, hidden*(K+1)]
        X = self.omega(z_flat)  # [B*N*T, C_out]
        X = X.view(batch_size, nodes_per_graph, -1)  # [B, N*T, C_out]
        
        # Extract last time slice from each graph
        outputs = []
        for b in range(batch_size):
            start = (T - 1) * N
            end = T * N
            y_hat_single = X[b, start:end, :]  # [N, C_out]
            outputs.append(y_hat_single)
        
        # Concatenate batch results: [B*N, C_out]
        y_hat = torch.cat(outputs, dim=0)
        
        return y_hat
