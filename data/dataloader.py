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


def get_dataloaders(input_vars, target_var, config,
                    batch_size=4, num_workers=0,
                    load_into_ram=False, device=None):
    """
    Create train/val/test dataloaders from local cached Zarr splits.
    """

    if device is None:
        device = torch.device("cpu")

    collate = gpu_collate_fn(device)

    # Make dataset objects
    train_ds = ERA5Dataset("train", input_vars, target_var, config,
                           load_into_ram=load_into_ram)
    val_ds   = ERA5Dataset("val", input_vars, target_var, config,
                           load_into_ram=load_into_ram)
    test_ds  = ERA5Dataset("test", input_vars, target_var, config,
                           load_into_ram=load_into_ram)

    print(f"Dataset sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Wrap in DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate
    )

    return train_loader, val_loader, test_loader
