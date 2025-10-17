#!/usr/bin/env python3
"""
Test SpatiotemporalSampler with Real ERA5 Data

This script tests the SpatiotemporalSampler by:
1. Loading a small subset of real ERA5 data
2. Creating temporal sequences
3. Testing spatial sampling
4. Validating batch shapes and data integrity
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import xarray as xr
from pathlib import Path
import yaml
from models.spatiotemporal_sampler import SpatiotemporalSampler


def load_era5_sample():
    """Load a small sample of real ERA5 data with all features (like dataloader)."""
    print("Loading ERA5 data sample with all features...")
    
    # Load config to get data paths
    with open("utils/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config["data"]["local_dir"])
    data_paths = config["data"]["data_paths"]
    levels = config["data"]["levels"]
    
    print(f"✓ Found {len(levels)} variables: {levels}")
    
    # Check if data files exist (check for any .nc files in each directory)
    missing_dirs = []
    for path_pattern in data_paths:
        dir_path = Path(path_pattern).parent
        if not dir_path.exists() or not list(dir_path.glob("*.nc")):
            missing_dirs.append(path_pattern)
    
    if missing_dirs:
        print(f"❌ Missing data files:")
        for f in missing_dirs:
            print(f"  {f}")
        print("Please run scripts/download_data.py first")
        return None
    
    # Load all variables (same as ERA5Dataset)
    all_data = []
    norm_stats = {}
    
    for path_pattern, level in zip(data_paths, levels):
        print(f"  Loading {level} from {path_pattern}...")
        
        # Open dataset for this variable
        ds = xr.open_mfdataset(path_pattern, combine="by_coords")
        
        # Take a small temporal subset (first 10 timesteps)
        ds_subset = ds.isel(time=slice(0, 10))
        
        # Compute normalization stats from full dataset (like ERA5Dataset)
        global_ds = xr.open_mfdataset(path_pattern, combine="by_coords")
        global_ds = global_ds.sel(time=slice("2010", "2018"))
        max_val = global_ds.max()[level].values
        min_val = global_ds.min()[level].values
        norm_stats[level] = (float(min_val), float(max_val))
        
        # Normalize to [0, 1]
        arr = (ds_subset[level] - min_val) / (max_val - min_val)
        arr = arr.load().values  # shape: (time, lat, lon)
        
        # Add channel dimension (time, 1, H, W)
        arr = torch.from_numpy(arr).float().unsqueeze(1)
        all_data.append(arr)
        
        print(f"    Shape: {arr.shape}")
    
    # Stack variables into a single tensor (time, channels, H, W)
    data_tensor = torch.cat(all_data, dim=1)
    
    # Cache dimensions
    T, C, H, W = data_tensor.shape
    
    print(f"✓ Combined tensor:")
    print(f"  Shape: {data_tensor.shape}")
    print(f"  Variables: {levels}")
    print(f"  Total features: {C}")
    print(f"  Time steps: {T}")
    print(f"  Spatial grid: {H} x {W}")
    
    # Add temporal (seasonal) encodings (like ERA5Dataset)
    print(f"  Adding temporal encodings...")
    ds_sample = xr.open_mfdataset(data_paths[0], combine="by_coords").isel(time=slice(0, T))
    temporal_features, _ = temporal_encoding(ds_sample["time"].values, H, W)
    data_tensor = torch.cat([data_tensor, temporal_features], dim=1)
    
    # Update channel count
    C_total = data_tensor.shape[1]
    
    print(f"✓ Final tensor with temporal encodings:")
    print(f"  Shape: {data_tensor.shape}")
    print(f"  Total features: {C_total} ({C} variables + 2 temporal)")
    
    # Convert to format expected by SpatiotemporalSampler: [T, H, W, features]
    data_tensor = data_tensor.permute(0, 2, 3, 1)  # [T, H, W, C_total]
    
    print(f"✓ Converted to sampler format: {data_tensor.shape}")
    
    # Print some statistics
    print(f"\n✓ Data statistics:")
    print(f"  Min value: {data_tensor.min().item():.3f}")
    print(f"  Max value: {data_tensor.max().item():.3f}")
    print(f"  Mean value: {data_tensor.mean().item():.3f}")
    print(f"  Std value: {data_tensor.std().item():.3f}")
    
    return data_tensor, T, H, W, levels, norm_stats


def temporal_encoding(times, H, W):
    """
    Compute cyclical temporal encodings for time-of-year information.
    Same as ERA5Dataset.temporal_encoding()
    """
    import pandas as pd
    
    times = np.array(times)
    
    # Compute day-of-year (1–365)
    doy = np.array([pd.Timestamp(t).dayofyear for t in times])
    
    # Cyclical encodings
    sin_doy = np.sin(2 * np.pi * doy / 365.0)
    cos_doy = np.cos(2 * np.pi * doy / 365.0)
    
    # from (time, 2) to (time, 2, H, W)
    temporal_features = np.stack([sin_doy, cos_doy], axis=1)
    temporal_features = (
        torch.from_numpy(temporal_features)
        .float()
        .unsqueeze(-1)          # add 1 new dimension, shape: (time, 2, 1)
        .unsqueeze(-1)          # add 1 new dimension, shape: (time, 2, 1, 1)
        .expand(-1, -1, H, W)   # broadcast those 1x1 spatial dims, shape: (time, 2, H, W)
    )
    
    return temporal_features, times


def test_sampler_initialization():
    """Test that SpatiotemporalSampler initializes correctly."""
    print("\n" + "="*50)
    print("TESTING SAMPLER INITIALIZATION")
    print("="*50)
    
    try:
        # Test with different configurations
        configs = [
            {"H": 8, "W": 8, "num_neighbors": [4, 2], "neighborhood": 4},
            {"H": 16, "W": 32, "num_neighbors": [8, 4], "neighborhood": 8},
            {"H": 4, "W": 4, "num_neighbors": [2], "neighborhood": 4}
        ]
        
        for i, config in enumerate(configs):
            print(f"\nTest {i+1}: H={config['H']}, W={config['W']}, neighbors={config['num_neighbors']}")
            
            sampler = SpatiotemporalSampler(**config)
            print(f"✓ Sampler created successfully")
            print(f"  Grid: {sampler.H} x {sampler.W} = {sampler.num_nodes} nodes")
            print(f"  Spatial edges: {sampler.spatial_edges.shape[1]} edges")
            
        return True
        
    except Exception as e:
        print(f"✗ Sampler initialization failed: {e}")
        return False


def test_sampling_with_real_data():
    """Test sampling with real ERA5 data."""
    print("\n" + "="*50)
    print("TESTING SAMPLING WITH REAL DATA")
    print("="*50)
    
    # Load real data
    result = load_era5_sample()
    if result is None:
        return False
    
    data_tensor, T, H, W, levels, norm_stats = result
    
    try:
        # Create sampler
        sampler = SpatiotemporalSampler(
            H=H, W=W, 
            num_neighbors=[8, 4], 
            neighborhood=4
        )
        print(f"✓ Created sampler for {H}x{W} grid")
        
        # Test single sampling
        print(f"\nTesting single sampling...")
        temporal_sequences, sampled_edges, sampled_nodes = sampler.sample_spatiotemporal(
            data_tensor, 
            batch_size=16,
            reference_timestep=0
        )
        
        print(f"✓ Single sampling successful:")
        print(f"  Temporal sequences: {temporal_sequences.shape}")
        print(f"  Sampled edges: {sampled_edges.shape}")
        print(f"  Sampled nodes: {sampled_nodes.shape}")
        print(f"  Number of sampled nodes: {len(sampled_nodes)}")
        print(f"  Features: {levels} + temporal encodings")
        
        # Validate shapes
        num_features = len(levels) + 2  # 5 variables + 2 temporal encodings
        expected_temporal_shape = (T, len(sampled_nodes), num_features)
        assert temporal_sequences.shape == expected_temporal_shape, \
            f"Expected {expected_temporal_shape}, got {temporal_sequences.shape}"
        
        assert sampled_edges.shape[0] == 2, f"Expected edge_index shape [2, E], got {sampled_edges.shape}"
        assert len(sampled_nodes) > 0, "No nodes were sampled"
        assert torch.all(sampled_nodes >= 0) and torch.all(sampled_nodes < H*W), \
            "Sampled node indices out of range"
        
        print(f"✓ All shape validations passed")
        
        # Test data integrity
        print(f"\nTesting data integrity...")
        
        # Check that temporal sequences contain actual data
        assert not torch.isnan(temporal_sequences).any(), "NaN values found in temporal sequences"
        assert not torch.isinf(temporal_sequences).any(), "Inf values found in temporal sequences"
        
        # Check that sampled nodes correspond to actual data
        for i, node_idx in enumerate(sampled_nodes):
            original_data = data_tensor[:, node_idx // W, node_idx % W, :]
            sampled_data = temporal_sequences[:, i, :]
            assert torch.allclose(original_data, sampled_data), \
                f"Data mismatch for node {node_idx}"
        
        print(f"✓ Data integrity checks passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Sampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_loader():
    """Test the batch loader functionality."""
    print("\n" + "="*50)
    print("TESTING BATCH LOADER")
    print("="*50)
    
    # Load real data
    result = load_era5_sample()
    if result is None:
        return False
    
    data_tensor, T, H, W, levels, norm_stats = result
    
    try:
        # Create sampler
        sampler = SpatiotemporalSampler(
            H=H, W=W, 
            num_neighbors=[6, 3], 
            neighborhood=4
        )
        
        print(f"✓ Created sampler for {H}x{W} grid")
        
        # Test batch loader
        print(f"\nTesting batch loader...")
        batch_size = 8
        num_batches = 0
        total_nodes_sampled = 0
        
        loader = sampler.get_spatiotemporal_loader(
            data_tensor, 
            batch_size=batch_size,
            reference_timestep=0
        )
        
        for batch_idx, (temporal_sequences, sampled_edges, sampled_nodes) in enumerate(loader):
            num_batches += 1
            total_nodes_sampled += len(sampled_nodes)
            
            print(f"  Batch {batch_idx + 1}:")
            print(f"    Temporal sequences: {temporal_sequences.shape}")
            print(f"    Sampled edges: {sampled_edges.shape}")
            print(f"    Sampled nodes: {len(sampled_nodes)} nodes")
            
            # Validate batch
            num_features = len(levels) + 2  # 5 variables + 2 temporal encodings
            assert temporal_sequences.shape[0] == T, f"Expected T={T}, got {temporal_sequences.shape[0]}"
            assert temporal_sequences.shape[2] == num_features, f"Expected {num_features} features, got {temporal_sequences.shape[2]}"
            assert len(sampled_nodes) > 0, "Empty batch"
            
            # Stop after a few batches for testing
            if batch_idx >= 2:
                break
        
        print(f"✓ Batch loader test successful:")
        print(f"  Processed {num_batches} batches")
        print(f"  Total nodes sampled: {total_nodes_sampled}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*50)
    print("TESTING EDGE CASES")
    print("="*50)
    
    try:
        # Test with very small grid
        print(f"\nTesting very small grid (2x2)...")
        small_data = torch.randn(3, 2, 2, 7)  # [T=3, H=2, W=2, features=7] (5 variables + 2 temporal)
        sampler = SpatiotemporalSampler(H=2, W=2, num_neighbors=[1], neighborhood=4)
        
        temporal_sequences, sampled_edges, sampled_nodes = sampler.sample_spatiotemporal(
            small_data, batch_size=1
        )
        
        print(f"✓ Small grid test passed: {temporal_sequences.shape}")
        
        # Test with invalid reference timestep
        print(f"\nTesting invalid reference timestep...")
        try:
            sampler.sample_spatiotemporal(small_data, reference_timestep=10)
            print(f"✗ Should have failed with invalid timestep")
            return False
        except AssertionError as e:
            print(f"✓ Correctly caught invalid timestep: {e}")
        
        # Test with wrong grid dimensions
        print(f"\nTesting wrong grid dimensions...")
        try:
            sampler.sample_spatiotemporal(torch.randn(3, 4, 4, 7))  # Wrong H, W
            print(f"✗ Should have failed with wrong dimensions")
            return False
        except AssertionError as e:
            print(f"✓ Correctly caught wrong dimensions: {e}")
        
        print(f"✓ All edge case tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("SPATIOTEMPORAL SAMPLER TEST SUITE")
    print("="*60)
    
    tests = [
        ("Sampler Initialization", test_sampler_initialization),
        ("Sampling with Real Data", test_sampling_with_real_data),
        ("Batch Loader", test_batch_loader),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! SpatiotemporalSampler is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)