import torch
from torch_geometric.loader import DataLoader   
from .dataset import ERA5Dataset


def get_dataloaders(config, batch_size=4, num_workers=0):
    """
    Create train/val/test dataloaders for ERA5 spatio-temporal graphs.
    """
    print("Creating datasets...")

    train_ds = ERA5Dataset(config["data_paths"], config["levels"], "train", config["time"])
    val_ds   = ERA5Dataset(config["data_paths"], config["levels"], "val",   config["time"])
    test_ds  = ERA5Dataset(config["data_paths"], config["levels"], "test",  config["time"])

    print(f"Dataset sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    print("Creating dataloaders...")

    train_loader = DataLoader(train_ds, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers)

    print("Dataloaders ready!")

    return train_loader, val_loader, test_loader
