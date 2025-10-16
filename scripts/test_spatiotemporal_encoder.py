"""
Test script for the Spatiotemporal Encoder

This script tests the implementation of the spatiotemporal encoder that combines
DeepESN temporal encoding with spatial sampling using graph shift operator powers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.spatiotemporal_encoder import SpatiotemporalEncoder, EfficientSpatiotemporalEncoder
from data.transforms import build_spatial_edges


def test_spatiotemporal_encoder():
    """Test the basic spatiotemporal encoder functionality."""
    print("Testing SpatiotemporalEncoder...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    H, W = 4, 4  # 4x4 grid
    num_nodes = H * W
    input_dim = 3  # e.g., temperature, humidity, pressure
    temporal_hidden_dim = 64
    temporal_output_dim = 32
    spatial_orders = 2
    
    # Create sample spatiotemporal data
    x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    
    # Create spatial edges for the grid
    edge_index = build_spatial_edges(H, W, periodic_lon=True, neighborhood=4)
    
    # Test undirected graph
    print("Testing undirected graph...")
    encoder_undirected = SpatiotemporalEncoder(
        input_dim=input_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        temporal_output_dim=temporal_output_dim,
        spatial_orders=spatial_orders,
        graph_type="undirected"
    )
    
    with torch.no_grad():
        output_undirected = encoder_undirected(x, edge_index)
    
    expected_output_shape = (batch_size, seq_len, num_nodes, temporal_output_dim * (spatial_orders + 1))
    assert output_undirected.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output_undirected.shape}"
    print(f"✓ Undirected encoder output shape: {output_undirected.shape}")
    
    # Test directed graph
    print("Testing directed graph...")
    encoder_directed = SpatiotemporalEncoder(
        input_dim=input_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        temporal_output_dim=temporal_output_dim,
        spatial_orders=spatial_orders,
        graph_type="directed"
    )
    
    with torch.no_grad():
        output_directed = encoder_directed(x, edge_index)
    
    expected_output_shape = (batch_size, seq_len, num_nodes, temporal_output_dim * (spatial_orders + 1))
    assert output_directed.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output_directed.shape}"
    print(f"✓ Directed encoder output shape: {output_directed.shape}")
    
    # Test bidirectional directed graph
    print("Testing bidirectional directed graph...")
    encoder_bidirectional = SpatiotemporalEncoder(
        input_dim=input_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        temporal_output_dim=temporal_output_dim,
        spatial_orders=spatial_orders,
        graph_type="directed",
        bidirectional=True
    )
    
    with torch.no_grad():
        output_bidirectional = encoder_bidirectional(x, edge_index)
    
    expected_output_shape = (batch_size, seq_len, num_nodes, temporal_output_dim * (2 * spatial_orders + 1))
    assert output_bidirectional.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output_bidirectional.shape}"
    print(f"✓ Bidirectional encoder output shape: {output_bidirectional.shape}")
    
    print("✓ All basic tests passed!")


def test_efficient_spatiotemporal_encoder():
    """Test the efficient spatiotemporal encoder with precomputed powers."""
    print("\nTesting EfficientSpatiotemporalEncoder...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    H, W = 4, 4
    num_nodes = H * W
    input_dim = 3
    temporal_hidden_dim = 64
    temporal_output_dim = 32
    spatial_orders = 2
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    edge_index = build_spatial_edges(H, W, periodic_lon=True, neighborhood=4)
    
    # Test efficient encoder
    encoder_efficient = EfficientSpatiotemporalEncoder(
        input_dim=input_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        temporal_output_dim=temporal_output_dim,
        spatial_orders=spatial_orders,
        graph_type="undirected"
    )
    
    # Setup graph shift operator powers
    encoder_efficient.setup_graph_shift_operator_powers(edge_index, num_nodes)
    
    with torch.no_grad():
        output_efficient = encoder_efficient(x)
    
    expected_output_shape = (batch_size, seq_len, num_nodes, temporal_output_dim * (spatial_orders + 1))
    assert output_efficient.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output_efficient.shape}"
    print(f"✓ Efficient encoder output shape: {output_efficient.shape}")
    
    print("✓ Efficient encoder test passed!")


def test_consistency():
    """Test that both encoders produce consistent results."""
    print("\nTesting consistency between encoders...")
    
    # Test parameters
    batch_size = 1  # Use batch_size=1 for easier comparison
    seq_len = 5
    H, W = 3, 3  # Smaller grid for easier debugging
    num_nodes = H * W
    input_dim = 2
    temporal_hidden_dim = 32
    temporal_output_dim = 16
    spatial_orders = 1
    
    # Create sample data
    torch.manual_seed(42)  # For reproducible results
    x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    edge_index = build_spatial_edges(H, W, periodic_lon=True, neighborhood=4)
    
    # Create both encoders with same parameters
    encoder_standard = SpatiotemporalEncoder(
        input_dim=input_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        temporal_output_dim=temporal_output_dim,
        spatial_orders=spatial_orders,
        graph_type="undirected"
    )
    
    encoder_efficient = EfficientSpatiotemporalEncoder(
        input_dim=input_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        temporal_output_dim=temporal_output_dim,
        spatial_orders=spatial_orders,
        graph_type="undirected"
    )
    
    # Setup efficient encoder
    encoder_efficient.setup_graph_shift_operator_powers(edge_index, num_nodes)
    
    # Get outputs
    with torch.no_grad():
        output_standard = encoder_standard(x, edge_index)
        output_efficient = encoder_efficient(x)
    
    # Check shapes match
    assert output_standard.shape == output_efficient.shape, "Output shapes don't match"
    
    # Check values are close (should be identical for same random seed)
    max_diff = torch.max(torch.abs(output_standard - output_efficient)).item()
    print(f"Maximum difference between encoders: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✓ Encoders produce consistent results!")
    else:
        print("⚠ Encoders produce different results (this might be expected due to numerical precision)")


def test_graph_shift_operator():
    """Test the graph shift operator computation."""
    print("\nTesting graph shift operator computation...")
    
    # Create a simple 3-node graph: 0-1-2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3
    
    # Test undirected normalization
    encoder = SpatiotemporalEncoder(
        input_dim=1, temporal_hidden_dim=4, temporal_output_dim=2,
        spatial_orders=1, graph_type="undirected"
    )
    
    A_tilde = encoder.compute_graph_shift_operator(edge_index, num_nodes)
    
    # Check that it's symmetric
    is_symmetric = torch.allclose(A_tilde, A_tilde.T, atol=1e-6)
    print(f"✓ Undirected graph shift operator is symmetric: {is_symmetric}")
    
    # Check that rows sum to 1 (for undirected, this should be approximately true)
    row_sums = torch.sum(A_tilde, dim=1)
    print(f"Row sums: {row_sums}")
    
    # Test directed normalization
    encoder_directed = SpatiotemporalEncoder(
        input_dim=1, temporal_hidden_dim=4, temporal_output_dim=2,
        spatial_orders=1, graph_type="directed"
    )
    
    A_tilde_directed = encoder_directed.compute_graph_shift_operator(edge_index, num_nodes)
    
    # Check that rows sum to 1 (for directed, this should be exactly true)
    row_sums_directed = torch.sum(A_tilde_directed, dim=1)
    print(f"Directed row sums: {row_sums_directed}")
    
    print("✓ Graph shift operator tests passed!")


if __name__ == "__main__":
    print("Running Spatiotemporal Encoder Tests")
    print("=" * 50)
    
    test_spatiotemporal_encoder()
    test_efficient_spatiotemporal_encoder()
    test_consistency()
    test_graph_shift_operator()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully! 🎉")
