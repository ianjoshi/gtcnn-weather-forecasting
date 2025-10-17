"""
Scalable GTCNN: Enhanced GTCNN with Spatial Sampling

This module wraps the existing GTCNN with spatial sampling to improve scalability.
The GTCNN architecture remains unchanged, but we feed it sampled subgraphs instead of full graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from models.gtcnn import GTCNN
from models.spatiotemporal_sampler import SpatiotemporalSampler


class ScalableGTCNN(nn.Module):
    """
    Scalable wrapper around GTCNN that uses spatial sampling for memory efficiency.
    
    This enhances the existing GTCNN by:
    1. Sampling spatial subgraphs instead of processing full graphs
    2. Maintaining the same input/output format as GTCNN
    3. Enabling training on much larger grids
    """
    
    def __init__(
        self,
        gtcnn: GTCNN,
        H: int,
        W: int,
        num_neighbors: list = [25, 10],
        neighborhood: int = 4,
        sampling_strategy: str = "single"  # "single", "multiple", "adaptive"
    ):
        """
        Initialize ScalableGTCNN.
        
        Args:
            gtcnn: The base GTCNN model
            H, W: Grid dimensions
            num_neighbors: Neighbor sampling configuration
            neighborhood: Spatial neighborhood size
            sampling_strategy: How to sample subgraphs
        """
        super().__init__()
        
        self.gtcnn = gtcnn
        self.H = H
        self.W = W
        self.num_nodes = H * W
        self.sampling_strategy = sampling_strategy
        
        # Create spatiotemporal sampler
        self.spatiotemporal_sampler = SpatiotemporalSampler(
            H=H,
            W=W,
            num_neighbors=num_neighbors,
            neighborhood=neighborhood
        )
        
        print(f"ScalableGTCNN initialized:")
        print(f"  Grid: {H} x {W} = {self.num_nodes} nodes")
        print(f"  Sampling strategy: {sampling_strategy}")
        print(f"  Neighbor sampling: {num_neighbors}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        N: int, 
        T: int,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Forward pass with spatial sampling.
        
        Args:
            x: [N*T, C_in] - flattened input (time then space)
            edge_index: [2, E] - spatial edges
            N: nodes per time slice (H*W)
            T: number of time steps
            num_samples: number of subgraphs to sample
            
        Returns:
            [N, C_out] - predictions for the last time slice
        """
        batch_size = N // self.num_nodes  # N is total nodes in batch, self.num_nodes is nodes per sample
        C_in = x.shape[1]
        
        # Validate input shapes
        assert x.shape[0] == batch_size * self.num_nodes * T, f"Input shape mismatch: expected {batch_size * self.num_nodes * T}, got {x.shape[0]}"
        assert N == batch_size * self.num_nodes, f"Node count mismatch: expected {batch_size * self.num_nodes}, got {N}"
        assert edge_index.shape[0] == 2, f"Expected edge_index shape [2, E], got {edge_index.shape}"
        
        # Reshape input to [batch_size, T, H*W, C_in]
        x_reshaped = x.view(batch_size, T, self.num_nodes, C_in)
        
        if self.sampling_strategy == "single":
            # Sample one subgraph per batch
            predictions = self._forward_single_sample(x_reshaped, edge_index, N, T)
        elif self.sampling_strategy == "multiple":
            # Sample multiple subgraphs and average predictions
            predictions = self._forward_multiple_samples(x_reshaped, edge_index, N, T, num_samples)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        return predictions
    
    def _forward_single_sample(
        self, 
        x_reshaped: torch.Tensor, 
        edge_index: torch.Tensor, 
        N: int, 
        T: int
    ) -> torch.Tensor:
        """Forward pass with single subgraph sampling."""
        batch_size, T, N_per_sample, C_in = x_reshaped.shape
        
        # Sample spatiotemporal subgraph from first batch item
        # Reshape from [T, H*W, C_in] to [T, H, W, C_in] for SpatiotemporalSampler
        sample_data = x_reshaped[0].view(T, self.H, self.W, C_in)
        temporal_sequences, sampled_edges, sampled_nodes = self.spatiotemporal_sampler.sample_spatiotemporal(
            sample_data,
            batch_size=1,
            reference_timestep=0
        )
        
        # Reshape for GTCNN: [sampled_nodes*T, C_in]
        sampled_x = temporal_sequences.view(-1, C_in)
        
        # Move sampled edges to the same device as the input
        sampled_edges = sampled_edges.to(x_reshaped.device)
        
        # Forward through GTCNN
        sampled_predictions = self.gtcnn(sampled_x, sampled_edges, len(sampled_nodes), T)
        
        # Map predictions back to full grid for all batch items
        full_predictions = torch.zeros(batch_size, N_per_sample, sampled_predictions.shape[1], device=x_reshaped.device)
        full_predictions[:, sampled_nodes] = sampled_predictions.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Flatten to match expected output format: [batch_size * N_per_sample, C_out]
        return full_predictions.view(-1, sampled_predictions.shape[1])
    
    def _forward_multiple_samples(
        self, 
        x_reshaped: torch.Tensor, 
        edge_index: torch.Tensor, 
        N: int, 
        T: int,
        num_samples: int
    ) -> torch.Tensor:
        """Forward pass with multiple subgraph sampling and averaging."""
        batch_size, T, N_per_sample, C_in = x_reshaped.shape
        
        all_predictions = []
        
        for _ in range(num_samples):
            # Sample spatiotemporal subgraph from first batch item
            # Reshape from [T, H*W, C_in] to [T, H, W, C_in] for SpatiotemporalSampler
            sample_data = x_reshaped[0].view(T, self.H, self.W, C_in)
            temporal_sequences, sampled_edges, sampled_nodes = self.spatiotemporal_sampler.sample_spatiotemporal(
                sample_data,
                batch_size=1,
                reference_timestep=0
            )
            
            # Reshape for GTCNN: [sampled_nodes*T, C_in]
            sampled_x = temporal_sequences.view(-1, C_in)
            
            # Move sampled edges to the same device as the input
            sampled_edges = sampled_edges.to(x_reshaped.device)
            
            # Forward through GTCNN
            sampled_predictions = self.gtcnn(sampled_x, sampled_edges, len(sampled_nodes), T)
            
            # Map to full grid for all batch items
            full_predictions = torch.zeros(batch_size, N_per_sample, sampled_predictions.shape[1], device=x_reshaped.device)
            full_predictions[:, sampled_nodes] = sampled_predictions.unsqueeze(0).expand(batch_size, -1, -1)
            all_predictions.append(full_predictions)
        
        # Average predictions from multiple samples
        averaged_predictions = torch.stack(all_predictions).mean(dim=0)
        
        # Flatten to match expected output format: [batch_size * N_per_sample, C_out]
        return averaged_predictions.view(-1, averaged_predictions.shape[-1])


