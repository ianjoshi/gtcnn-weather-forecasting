from torch.utils.data import DataLoader
from .dataset import ERA5Dataset

def get_dataloaders(xr_dataset, input_vars, target_var,
                    input_length=7, forecast_horizon=1,
                    batch_size=4, num_workers=0):
    """
    Create train/val/test dataloaders.
    """
    train_ds = ERA5Dataset(xr_dataset, input_vars, target_var, 
                           input_length, forecast_horizon, split="train")
    val_ds   = ERA5Dataset(xr_dataset, input_vars, target_var,
                           input_length, forecast_horizon, split="val")
    test_ds  = ERA5Dataset(xr_dataset, input_vars, target_var,
                           input_length, forecast_horizon, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader
