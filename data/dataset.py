import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from pathlib import Path

from data.transforms import *


class ERA5Dataset(Dataset):
    def __init__(self, data_paths, levels, split, config, drop_leap=True):
        """
        ERA5 dataset loader.

        Args:
            data_paths (list[str]): list of glob patterns for .nc files (per variable)
            levels (list[str]): variable names matching data_paths order
            split (str): "train", "val", or "test"
            config (dict): config["time"] dictionary with split ranges
            drop_leap (bool): drop Feb 29th for leap years
        """
        self.levels = levels
        self.drop_leap = drop_leap
        self.input_length = 7
        self.forecast_horizon = 1

        # Pick time range
        if split == "train":
            time_range = slice(config["train_start"], config["train_end"])
        elif split == "val":
            time_range = slice(config["val_start"], config["val_end"])
        elif split == "test":
            time_range = slice(config["test_start"], config["test_end"])
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Load and normalize each variable
        self.data = []
        for path, lev in zip(data_paths, levels):
            ds = xr.open_mfdataset(path, combine="by_coords")
            ds = ds.sel(time=time_range)

            # Global min/max 
            global_ds = xr.open_mfdataset(path, combine="by_coords")
            global_ds = global_ds.sel(time=slice("2005", "2018"))
            max_val = global_ds.max()[lev].values
            min_val = global_ds.min()[lev].values

            arr = (ds[lev] - min_val) / (max_val - min_val)
            arr = arr.load().values  # shape: (time, lat, lon)

            if drop_leap:
                arr = self._drop_leap_days(arr, ds["time"].values)

            # Reshape to (time, 1, H, W)
            arr = torch.from_numpy(arr).float().unsqueeze(1)
            self.data.append(arr)

        # Stack into (time, channels, H, W)
        self.data = torch.cat(self.data, dim=1)  # (time, channels, H, W)

        # Cache grid size
        _, self.C, self.H, self.W = self.data.shape

    def _drop_leap_days(self, arr, time):
        """Drop Feb 29 from leap years."""
        mask = np.array(
            [not (t.astype("datetime64[M]").astype(int) % 12 == 1
                  and t.astype("datetime64[D]").astype(int) % 29 == 0)
             for t in time]
        )
        return arr[mask]

    def __len__(self):
        # Each sample = input_length past steps → target
        return self.data.shape[0] - self.input_length - self.forecast_horizon

    def __getitem__(self, idx):
        start = idx
        end = idx + self.input_length
        target_idx = end + self.forecast_horizon - 1

        X = self.data[start:end]       # (input_length, channels, H, W)
        y = self.data[target_idx]      # (channels, H, W) or pick one channel later

        # Convert to spatio-temporal PyG graph
        graph = to_spatio_temporal_graph(X, y, self.H, self.W)
        return graph
