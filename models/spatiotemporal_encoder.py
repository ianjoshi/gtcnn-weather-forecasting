"""
Spatiotemporal Encoder Implementation

This module implements the complete spatiotemporal encoder from "Scalable Spatiotemporal Graph Neural Networks"
by Cini et al., combining:
1. Deep Echo State Network (DeepESN) for temporal encoding
2. Spatial sampling using graph shift operator powers

Implements equations (3), (4), and (5) from section 3.1 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from .temporal_encoder import RandomizedRNNEncoder


class SpatiotemporalEncoder(nn.Module):
    """
    Complete spatiotemporal encoder combining DeepESN temporal encoding with spatial sampling.
    
    This implements the full encoder described in section 3.1 of the SGP paper:
    - Temporal encoding using DeepESN (equations 3-4)
    - Spatial sampling using graph shift operator powers (equation 5)
    
    The encoder processes spatiotemporal data by:
    1. Encoding temporal dynamics at each node using DeepESN
    2. Propagating information spatially using powers of graph shift operator
    3. Concatenating multi-scale spatial representations
    """
    
    def __init__(
        self,
        input_dim: int,
        temporal_hidden_dim: int,
        temporal_output_dim: int,
        num_temporal_layers: int = 2,
        spatial_orders: int = 3,  # K in equation 5
        graph_type: str = "undirected",  # "directed" or "undirected"
        bidirectional: bool = False,  # For directed graphs, use bidirectional dynamics
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.temporal_output_dim = temporal_output_dim
        self.spatial_orders = spatial_orders
        self.graph_type = graph_type
        self.bidirectional = bidirectional
        
        # Temporal encoder (DeepESN)
        self.temporal_encoder = RandomizedRNNEncoder(
            input_dim=input_dim,
            hidden_dim=temporal_hidden_dim,
            output_dim=temporal_output_dim,
            num_layers=num_temporal_layers,
            dropout=dropout
        )
        
        # Output dimension after spatial sampling
        # For bidirectional directed graphs: 2K + 1 orders
        # For undirected graphs: K + 1 orders
        if graph_type == "directed" and bidirectional:
            self.spatial_orders_total = 2 * spatial_orders + 1
        else:
            self.spatial_orders_total = spatial_orders + 1
            
        self.output_dim = temporal_output_dim * self.spatial_orders_total
    
    def compute_graph_shift_operator(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the normalized graph shift operator Ã.
        
        For undirected graphs: Ã = D^(-1/2) A D^(-1/2) (symmetric normalization)
        For directed graphs: Ã = D^(-1) A (row normalization)
        
        Args:
            edge_index: [2, num_edges] - edge indices
            num_nodes: Number of nodes in the graph
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Ã: [num_nodes, num_nodes] - normalized adjacency matrix
        """
        # Create adjacency matrix
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Create sparse adjacency matrix
        A = torch.sparse_coo_tensor(
            edge_index, 
            edge_weight, 
            size=(num_nodes, num_nodes),
            device=edge_index.device
        )
        
        # Compute degree matrix
        if self.graph_type == "undirected":
            # Symmetric normalization: D^(-1/2) A D^(-1/2)
            degrees = torch.sparse.sum(A, dim=1).to_dense()
            degrees_inv_sqrt = torch.pow(degrees + 1e-8, -0.5)  # Add small epsilon to avoid division by zero
            degrees_inv_sqrt[degrees == 0] = 0  # Handle isolated nodes
            
            # Create diagonal matrix
            D_inv_sqrt = torch.diag(degrees_inv_sqrt)
            A_tilde = D_inv_sqrt @ A.to_dense() @ D_inv_sqrt
            
        else:  # directed
            # Row normalization: D^(-1) A
            degrees = torch.sparse.sum(A, dim=1).to_dense()
            degrees_inv = torch.pow(degrees + 1e-8, -1.0)  # Add small epsilon to avoid division by zero
            degrees_inv[degrees == 0] = 0  # Handle isolated nodes
            
            # Create diagonal matrix
            D_inv = torch.diag(degrees_inv)
            A_tilde = D_inv @ A.to_dense()
        
        return A_tilde
    
    def spatial_sampling(
        self, 
        H: torch.Tensor, 
        A_tilde: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform spatial sampling using powers of graph shift operator.
        
        Implements equation (5) from the paper:
        S^(0)_t = H_t = [H^(0)_t || H^(1)_t || ... || H^(L)_t]
        S^(k)_t = Ã S^(k-1)_t = [Ã^k H^(0)_t || Ã^k H^(1)_t || ... || Ã^k H^(L)_t]
        S_t = [S^(0)_t || S^(1)_t || ... || S^(K)_t]
        
        Args:
            H: [batch_size, seq_len, num_nodes, temporal_output_dim] - temporal encodings
            A_tilde: [num_nodes, num_nodes] - normalized adjacency matrix
            
        Returns:
            S: [batch_size, seq_len, num_nodes, output_dim] - spatiotemporal encodings
        """
        batch_size, seq_len, num_nodes, temporal_dim = H.shape
        
        # Initialize spatial representations
        spatial_reprs = []
        
        # S^(0)_t = H_t (no spatial propagation)
        S_0 = H  # [batch_size, seq_len, num_nodes, temporal_output_dim]
        spatial_reprs.append(S_0)
        
        # Compute spatial orders k = 1, ..., K
        S_prev = H
        for k in range(1, self.spatial_orders + 1):
            # S^(k)_t = Ã S^(k-1)_t
            # Reshape for matrix multiplication: [batch_size * seq_len, num_nodes, temporal_output_dim]
            S_prev_flat = S_prev.view(batch_size * seq_len, num_nodes, temporal_dim)
            
            # Apply graph shift operator: Ã @ S^(k-1)_t
            S_k_flat = torch.matmul(A_tilde, S_prev_flat)  # [batch_size * seq_len, num_nodes, temporal_output_dim]
            
            # Reshape back: [batch_size, seq_len, num_nodes, temporal_output_dim]
            S_k = S_k_flat.view(batch_size, seq_len, num_nodes, temporal_dim)
            spatial_reprs.append(S_k)
            
            # Update for next iteration
            S_prev = S_k
        
        # For directed graphs with bidirectional dynamics, also compute with Ã^T
        if self.graph_type == "directed" and self.bidirectional:
            S_prev = H
            for k in range(1, self.spatial_orders + 1):
                # S^(k)_t = Ã^T S^(k-1)_t (backward propagation)
                S_prev_flat = S_prev.view(batch_size * seq_len, num_nodes, temporal_dim)
                S_k_flat = torch.matmul(A_tilde.T, S_prev_flat)
                S_k = S_k_flat.view(batch_size, seq_len, num_nodes, temporal_dim)
                spatial_reprs.append(S_k)
                S_prev = S_k
        
        # Concatenate all spatial representations
        # S_t = [S^(0)_t || S^(1)_t || ... || S^(K)_t] (and possibly backward orders)
        S = torch.cat(spatial_reprs, dim=-1)  # [batch_size, seq_len, num_nodes, output_dim]
        
        return S
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the spatiotemporal encoder.
        
        Args:
            x: [batch_size, seq_len, num_nodes, input_dim] - input spatiotemporal data
            edge_index: [2, num_edges] - graph edge indices
            edge_weight: Optional [num_edges] - edge weights
            
        Returns:
            [batch_size, seq_len, num_nodes, output_dim] - spatiotemporal encodings
        """
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # Step 1: Temporal encoding at each node
        # Reshape for temporal encoder: [batch_size * num_nodes, seq_len, input_dim]
        x_reshaped = x.view(batch_size * num_nodes, seq_len, input_dim)
        
        # Apply temporal encoder: [batch_size * num_nodes, seq_len, temporal_output_dim]
        H_flat = self.temporal_encoder(x_reshaped)
        
        # Reshape back: [batch_size, seq_len, num_nodes, temporal_output_dim]
        H = H_flat.view(batch_size, seq_len, num_nodes, self.temporal_output_dim)
        
        # Step 2: Compute graph shift operator
        A_tilde = self.compute_graph_shift_operator(edge_index, num_nodes, edge_weight)
        
        # Step 3: Spatial sampling using powers of graph shift operator
        S = self.spatial_sampling(H, A_tilde)
        
        return S


class EfficientSpatiotemporalEncoder(nn.Module):
    """
    Memory-efficient version of the spatiotemporal encoder.
    
    This version precomputes powers of the graph shift operator to avoid
    repeated matrix multiplications during training, as suggested in the paper.
    """
    
    def __init__(
        self,
        input_dim: int,
        temporal_hidden_dim: int,
        temporal_output_dim: int,
        num_temporal_layers: int = 2,
        spatial_orders: int = 3,
        graph_type: str = "undirected",
        bidirectional: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.temporal_output_dim = temporal_output_dim
        self.spatial_orders = spatial_orders
        self.graph_type = graph_type
        self.bidirectional = bidirectional
        
        # Temporal encoder
        self.temporal_encoder = RandomizedRNNEncoder(
            input_dim=input_dim,
            hidden_dim=temporal_hidden_dim,
            output_dim=temporal_output_dim,
            num_layers=num_temporal_layers,
            dropout=dropout
        )
        
        # Precomputed graph shift operator powers (will be set during setup)
        self.register_buffer('A_tilde_powers', None)
        
        # Output dimension
        if graph_type == "directed" and bidirectional:
            self.spatial_orders_total = 2 * spatial_orders + 1
        else:
            self.spatial_orders_total = spatial_orders + 1
            
        self.output_dim = temporal_output_dim * self.spatial_orders_total
    
    def setup_graph_shift_operator_powers(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None
    ):
        """
        Precompute powers of the graph shift operator for efficient inference.
        
        This should be called once after the graph structure is known.
        """
        # Compute base graph shift operator
        A_tilde = self._compute_graph_shift_operator(edge_index, num_nodes, edge_weight)
        
        # Precompute powers Ã^0, Ã^1, ..., Ã^K
        A_powers = [torch.eye(num_nodes, device=A_tilde.device)]  # Ã^0 = I
        
        A_current = A_tilde
        for k in range(1, self.spatial_orders + 1):
            A_powers.append(A_current)
            A_current = A_current @ A_tilde
        
        # For bidirectional directed graphs, also compute powers of Ã^T
        if self.graph_type == "directed" and self.bidirectional:
            A_T_current = A_tilde.T
            for k in range(1, self.spatial_orders + 1):
                A_powers.append(A_T_current)
                A_T_current = A_T_current @ A_tilde.T
        
        # Store as buffer
        self.A_tilde_powers = torch.stack(A_powers, dim=0)  # [num_powers, num_nodes, num_nodes]
    
    def _compute_graph_shift_operator(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Same as SpatiotemporalEncoder.compute_graph_shift_operator"""
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        A = torch.sparse_coo_tensor(
            edge_index, 
            edge_weight, 
            size=(num_nodes, num_nodes),
            device=edge_index.device
        )
        
        if self.graph_type == "undirected":
            degrees = torch.sparse.sum(A, dim=1).to_dense()
            degrees_inv_sqrt = torch.pow(degrees + 1e-8, -0.5)
            degrees_inv_sqrt[degrees == 0] = 0
            D_inv_sqrt = torch.diag(degrees_inv_sqrt)
            A_tilde = D_inv_sqrt @ A.to_dense() @ D_inv_sqrt
        else:
            degrees = torch.sparse.sum(A, dim=1).to_dense()
            degrees_inv = torch.pow(degrees + 1e-8, -1.0)
            degrees_inv[degrees == 0] = 0
            D_inv = torch.diag(degrees_inv)
            A_tilde = D_inv @ A.to_dense()
        
        return A_tilde
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using precomputed graph shift operator powers.
        
        Args:
            x: [batch_size, seq_len, num_nodes, input_dim] - input spatiotemporal data
            
        Returns:
            [batch_size, seq_len, num_nodes, output_dim] - spatiotemporal encodings
        """
        if self.A_tilde_powers is None:
            raise ValueError("Graph shift operator powers not set. Call setup_graph_shift_operator_powers first.")
        
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # Temporal encoding
        x_reshaped = x.view(batch_size * num_nodes, seq_len, input_dim)
        H_flat = self.temporal_encoder(x_reshaped)
        H = H_flat.view(batch_size, seq_len, num_nodes, self.temporal_output_dim)
        
        # Spatial sampling using precomputed powers
        spatial_reprs = []
        for k, A_power in enumerate(self.A_tilde_powers):
            # Apply Ã^k to temporal encodings
            H_flat = H.view(batch_size * seq_len, num_nodes, self.temporal_output_dim)
            S_k_flat = torch.matmul(A_power, H_flat)
            S_k = S_k_flat.view(batch_size, seq_len, num_nodes, self.temporal_output_dim)
            spatial_reprs.append(S_k)
        
        # Concatenate all spatial representations
        S = torch.cat(spatial_reprs, dim=-1)
        
        return S
