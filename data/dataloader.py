import torch
from torch_geometric.loader import DataLoader   
from .dataset import ERA5Dataset


def get_datasets(config, model_type, eval_mode=False):
    """
    Create ERA5 train/val/test datasets.

    Args:
        config (dict): Configuration dictionary with keys:
            - config["data"]["data_paths"]: list of NetCDF file paths (one per variable)
            - config["data"]["levels"]: list of variable names (same order as data_paths)
            - config["data"]["time"]: dict with split ranges
            - config["data"]["sample_rate"]: daily or hourly
            - config["graph"]["input_length"]: # past timesteps for input
            - config["graph"]["forecast_horizon"]: # steps ahead for prediction
            - config["graph"]["neighborhood"]: 4 or 8 (spatial neighbors)
            - config["graph"]["type"]: "cartesian" or "strong"

    Returns:
        (train_ds, val_ds, test_ds): ERA5Dataset objects
    """
    print("Creating datasets...")

    # Extract config fields 
    data_paths = config["data"]["data_paths"] 
    levels = config["data"]["levels"]
    time_slices = config["data"]["time"]
    sample_rate = config["data"]["sample_rate"]

    input_length = config["graph"]["input_length"]
    forecast_horizon = config["graph"]["forecast_horizon"]
    neighborhood = config["graph"]["neighborhood"]
    graph_type = config["graph"]["type"]

    if model_type in ["cnn3d", "convlstm"]:
        graph_mode = False
    else:
        graph_mode = True
        print("Using a graph model...")

    # Build datasets
    if not eval_mode:
        train_ds = ERA5Dataset(
            split="train",
            data_paths=data_paths, levels=levels, time_slices=time_slices,
            sample_rate=sample_rate, input_length=input_length, 
            forecast_horizon=forecast_horizon, neighborhood=neighborhood, 
            graph_type=graph_type, graph_mode=graph_mode
        )
        val_ds = ERA5Dataset(
            split="val",
            data_paths=data_paths, levels=levels, time_slices=time_slices,
            sample_rate=sample_rate, input_length=input_length, 
            forecast_horizon=forecast_horizon, neighborhood=neighborhood, 
            graph_type=graph_type, graph_mode=graph_mode
        )
        print(f"Dataset sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds
    else:
        test_ds = ERA5Dataset(
            split="test",
            data_paths=data_paths, levels=levels, time_slices=time_slices,
            sample_rate=sample_rate, input_length=input_length, 
            forecast_horizon=forecast_horizon, neighborhood=neighborhood, 
            graph_type=graph_type, graph_mode=graph_mode
        )
        print(f"Dataset sizes -> Test: {len(test_ds)}")
        return test_ds


def get_dataloaders(config, model_type, eval_mode=False):
    """
    Wrap ERA5 datasets in PyTorch Geometric DataLoaders.

    Args:
        config (dict): Configuration dictionary with keys:
            - config["training"]["batch_size"]: mini-batch size
            - config["training"]["num_workers"]: number of dataloader workers
            plus the dataset config keys described in get_datasets.

    Returns:
        (train_loader, val_loader, test_loader): torch_geometric DataLoader objects
    """
    # Extract dataloader config 
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    # Load datasets 
    if not eval_mode:
        train_ds, val_ds = get_datasets(config=config, model_type=model_type, eval_mode=eval_mode)
        
        print("Creating dataloaders...")
        
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader
    else:
        test_ds = get_datasets(config=config, model_type=model_type, eval_mode=eval_mode)
        
        print("Creating dataloaders...")
        
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return test_loader
