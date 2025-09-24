# import torch
# from torch.utils.data import DataLoader
# from .dataset import ERA5Dataset


# def gpu_collate_fn(device):
#     def collate(batch):
#         # unpack first two items only
#         Xs, ys, *others = zip(*batch) if len(batch[0]) > 2 else zip(*batch)
#         Xs = torch.stack(Xs).to(device, non_blocking=True)
#         ys = torch.stack(ys).to(device, non_blocking=True)
#         return (Xs, ys) if not others else (Xs, ys, others)
#     return collate



# def get_dataloaders(config, batch_size=4, num_workers=0, device=None):
#     """
#     Create train/val/test dataloaders from local cached Zarr splits.
#     """

#     if device is None:
#         device = torch.device("cpu")

#     collate = gpu_collate_fn(device)

#     # Make dataset objects

#     print("Creating datasets...")

#     train_ds = ERA5Dataset(config["data_paths"], config["levels"], "train", config["time"])
#     val_ds   = ERA5Dataset(config["data_paths"], config["levels"], "val",   config["time"])
#     test_ds  = ERA5Dataset(config["data_paths"], config["levels"], "test",  config["time"])

#     print(f"Dataset sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

#     print("Creating dataloaders...")

#     # Wrap in DataLoaders
#     train_loader = DataLoader(
#         train_ds, batch_size=batch_size, shuffle=True,
#         num_workers=num_workers, collate_fn=collate
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, collate_fn=collate
#     )
#     test_loader = DataLoader(
#         test_ds, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, collate_fn=collate
#     )

#     print("Dataloaders ready!")

#     return train_loader, val_loader, test_loader


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
