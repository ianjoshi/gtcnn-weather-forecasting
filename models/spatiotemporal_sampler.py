"""
Spatiotemporal Sampler: Sample First, Then Add Temporal

This module implements the approach where we:
1. Sample spatial subgraphs using NeighborLoader
2. Extract temporal sequences for the sampled nodes
3. Combine spatial and temporal information efficiently
"""

import torch
import numpy as np
import xarray as xr
from pathlib import Path
import yaml
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from typing import List, Tuple, Optional
from data.transforms import build_spatial_edges


class SpatiotemporalSampler:
    """
    Spatiotemporal sampler that combines spatial sampling with temporal sequences.
    
    Approach:
    1. Sample spatial subgraph from reference timestep
    2. Extract temporal sequences for sampled nodes
    3. Return combined spatiotemporal data
    """
    
    def __init__(
        self,
        H: int,
        W: int,
        num_neighbors: List[int] = [25, 10],
        neighborhood: int = 4,
        periodic_lon: bool = True
    ):
        """
        Initialize the spatiotemporal sampler.
        
        Args:
            H, W: Grid dimensions
            num_neighbors: Number of neighbors to sample for each layer
            neighborhood: Neighborhood size for building spatial edges
            periodic_lon: Whether to use periodic longitude
        """
        self.H = H
        self.W = W
        self.num_nodes = H * W
        self.num_neighbors = num_neighbors
        
        # Build the static spatial graph
        self.spatial_edges = build_spatial_edges(H, W, periodic_lon, neighborhood)
        
        print(f"Built spatiotemporal sampler:")
        print(f"  Grid: {H} x {W} = {self.num_nodes} nodes")
        print(f"  Spatial edges: {self.spatial_edges.shape[1]}")
        print(f"  Neighbor sampling: {num_neighbors}")
    
    def sample_spatiotemporal(
        self, 
        temporal_data: torch.Tensor, 
        batch_size: int = 32,
        reference_timestep: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample spatiotemporal data: spatial subgraph + temporal sequences.
        
        Args:
            temporal_data: [T, H, W, features] - temporal data for all timesteps
            batch_size: Batch size for spatial sampling
            reference_timestep: Which timestep to use for spatial sampling
            
        Returns:
            temporal_sequences: [T, sampled_nodes, features] - temporal sequences for sampled nodes
            spatial_edges: [2, sampled_edges] - spatial edges in sampled subgraph
            sampled_node_indices: [sampled_nodes] - original node indices of sampled nodes
        """
        T, H, W, features = temporal_data.shape
        
        # Validate input shapes
        assert temporal_data.dim() == 4, f"Expected 4D tensor [T, H, W, features], got {temporal_data.dim()}D"
        assert H == self.H and W == self.W, f"Grid dimensions mismatch: expected ({self.H}, {self.W}), got ({H}, {W})"
        assert 0 <= reference_timestep < T, f"Reference timestep {reference_timestep} out of range [0, {T-1}]"
        
        # Step 1: Sample spatial subgraph from reference timestep
        reference_features = temporal_data[reference_timestep]  # [H, W, features]
        reference_flat = reference_features.view(self.num_nodes, features)  # [num_nodes, features]
        
        # Create spatial data for sampling
        spatial_data = Data(x=reference_flat, edge_index=self.spatial_edges)
        
        # Sample spatial subgraph
        loader = NeighborLoader(
            spatial_data,
            num_neighbors=self.num_neighbors,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False
        )
        
        sampled_batch = next(iter(loader))
        
        # Step 2: Get sampled node indices
        # NeighborLoader returns a batch with 'batch' attribute indicating which original nodes were sampled
        # We need to extract the unique node indices from the batch
        if hasattr(sampled_batch, 'batch') and sampled_batch.batch is not None:
            # Get unique node indices that were actually sampled
            sampled_node_indices = torch.unique(sampled_batch.batch)
        else:
            # Fallback: assume all nodes in the batch are sampled
            # This happens when NeighborLoader doesn't set the batch attribute
            sampled_node_indices = torch.arange(sampled_batch.x.shape[0], device=sampled_batch.x.device)
        
        sampled_edges = sampled_batch.edge_index     # [2, sampled_edges]
        
        # Step 3: Extract temporal sequences for sampled nodes
        # Reshape temporal data to [T, num_nodes, features]
        temporal_flat = temporal_data.view(T, self.num_nodes, features)
        
        # Extract sequences for sampled nodes
        temporal_sequences = temporal_flat[:, sampled_node_indices, :]  # [T, sampled_nodes, features]
        
        # Validate sampling results
        assert len(sampled_node_indices) > 0, "No nodes were sampled"
        assert torch.all(sampled_node_indices >= 0) and torch.all(sampled_node_indices < self.num_nodes), \
            f"Sampled node indices out of range [0, {self.num_nodes-1}]"
        assert temporal_sequences.shape == (T, len(sampled_node_indices), features), \
            f"Unexpected temporal_sequences shape: {temporal_sequences.shape}"
        assert sampled_edges.shape[0] == 2, f"Expected edge_index shape [2, E], got {sampled_edges.shape}"
        
        return temporal_sequences, sampled_edges, sampled_node_indices
    
    def get_spatiotemporal_loader(
        self, 
        temporal_data: torch.Tensor, 
        batch_size: int = 32,
        reference_timestep: int = 0
    ):
        """
        Create a loader that yields spatiotemporal batches.
        
        Args:
            temporal_data: [T, H, W, features] - temporal data for all timesteps
            batch_size: Batch size for spatial sampling
            reference_timestep: Which timestep to use for spatial sampling
            
        Yields:
            temporal_sequences: [T, sampled_nodes, features]
            spatial_edges: [2, sampled_edges]
            sampled_node_indices: [sampled_nodes]
        """
        T, H, W, features = temporal_data.shape
        
        # Create spatial data for sampling
        reference_features = temporal_data[reference_timestep]
        reference_flat = reference_features.view(self.num_nodes, features)
        spatial_data = Data(x=reference_flat, edge_index=self.spatial_edges)
        
        # Create spatial loader
        spatial_loader = NeighborLoader(
            spatial_data,
            num_neighbors=self.num_neighbors,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False
        )
        
        # Reshape temporal data for efficient indexing
        temporal_flat = temporal_data.view(T, self.num_nodes, features)
        
        # Yield spatiotemporal batches
        for spatial_batch in spatial_loader:
            if hasattr(spatial_batch, 'batch') and spatial_batch.batch is not None:
                # Get unique node indices that were actually sampled
                sampled_node_indices = torch.unique(spatial_batch.batch)
            else:
                # Fallback: assume all nodes in the batch are sampled
                sampled_node_indices = torch.arange(spatial_batch.x.shape[0], device=spatial_batch.x.device)
            
            sampled_edges = spatial_batch.edge_index
            
            # Extract temporal sequences for sampled nodes
            temporal_sequences = temporal_flat[:, sampled_node_indices, :]
            
            yield temporal_sequences, sampled_edges, sampled_node_indices

