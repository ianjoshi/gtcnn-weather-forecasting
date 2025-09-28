import torch
from torch_geometric.loader import DataLoader   
from .dataset import ERA5Dataset

def get_datasets(config):

    print("Creating datasets...")

    data_paths = config["data"]["data_paths"] 
    levels = config["data"]["levels"]
    time_slices = config["data"]["time"]

    input_length = config["graph"]["input_length"]
    forecast_horizon = config["graph"]["forecast_horizon"]
    neighborhood = config["graph"]["neighborhood"]

    train_ds = ERA5Dataset(split="train", data_paths=data_paths, levels=levels, 
                           time_slices=time_slices, input_length=input_length, 
                           forecast_horizon=forecast_horizon, neighborhood=neighborhood)
    val_ds = ERA5Dataset(split="val", data_paths=data_paths, levels=levels, 
                         time_slices=time_slices, input_length=input_length, 
                         forecast_horizon=forecast_horizon, neighborhood=neighborhood)
    test_ds = ERA5Dataset(split="test", data_paths=data_paths, levels=levels, 
                          time_slices=time_slices, input_length=input_length, 
                          forecast_horizon=forecast_horizon, neighborhood=neighborhood)

    print(f"Dataset sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return train_ds, val_ds, test_ds


def get_dataloaders(config):
    """
    Create train/val/test dataloaders for ERA5 spatio-temporal graphs.
    """
    
    train_ds, val_ds, test_ds = get_datasets(config=config)

    print("Creating dataloaders...")

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers)

    print("Dataloaders ready!")

    return train_loader, val_loader, test_loader
