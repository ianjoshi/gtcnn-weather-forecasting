import torch
from torch.utils.data import DataLoader
from .dataset import ERA5Dataset


def gpu_collate_fn(device):
    """
    Custom collate function that moves batch tensors to the given device.
    """
    def collate(batch):
        Xs, ys = zip(*batch)   # unpack list of (X, y) pairs
        Xs = torch.stack(Xs)
        ys = torch.stack(ys)
        return Xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
    return collate


def get_dataloaders(xr_dataset, input_vars, target_var, config,
                    input_length=7, forecast_horizon=1,
                    batch_size=4, num_workers=0,
                    device=None):
    """
    Create train/val/test dataloaders, optionally GPU-aware.
    """
    train_ds = ERA5Dataset(xr_dataset, input_vars, target_var, config,
                           input_length, forecast_horizon, split="train")
    val_ds   = ERA5Dataset(xr_dataset, input_vars, target_var, config,
                           input_length, forecast_horizon, split="val")
    test_ds  = ERA5Dataset(xr_dataset, input_vars, target_var, config,
                            input_length, forecast_horizon, split="test")
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    # Default: CPU device
    if device is None:
        device = torch.device("cpu")

    collate = gpu_collate_fn(device)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate
    )

    return train_loader, val_loader, test_loader
