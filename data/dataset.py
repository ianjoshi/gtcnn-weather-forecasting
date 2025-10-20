import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from pathlib import Path

from data.transforms import *


class ERA5Dataset(Dataset):
    def __init__(self, 
                 split, 
                 data_paths, 
                 levels, 
                 time_slices, 
                 sample_rate,
                 input_length, 
                 forecast_horizon, 
                 neighborhood,
                 graph_type,
                 graph_mode=True,
                 drop_leap=True):
        """
        ERA5 dataset loader for spatio-temporal forecasting.

        This class loads ERA5 reanalysis data from NetCDF files,
        normalizes each variable globally, and prepares samples 
        as PyTorch Geometric graph objects with spatio-temporal edges.

        Args:
            split (str): One of {"train", "val", "test"} defining the time range.
            data_paths (list[str]): List of glob patterns for NetCDF files, 
                one per variable (e.g., temperature, wind).
            levels (list[str]): Variable names (must match data_paths order).
            time_slices (dict): Dict with split ranges.
            daily_sample (str): Daily or Hourly.
            input_length (int): Number of past timesteps per input sequence.
            forecast_horizon (int): Prediction horizon in timesteps.
            neighborhood (int): Spatial neighborhood size (4 or 8).
            graph_type (str): Graph construction mode:
                - "cartesian": temporal self-links only
                - "strong": temporal links to spatial neighbors at next timestep
            graph_mode (bool, optional): if True, turn the dataset into a graph. 
            drop_leap (bool, optional): If True, drop Feb 29 from leap years.
        """
        self.levels = levels
        self.drop_leap = drop_leap
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.neighborhood = neighborhood
        self.graph_type = graph_type
        self.graph_mode = graph_mode

        self.norm_stats = {}

        # Select time range based on split
        if split == "train":
            time_range = slice(time_slices["train_start"], time_slices["train_end"])
        elif split == "val":
            time_range = slice(time_slices["val_start"], time_slices["val_end"])
        elif split == "test":
            time_range = slice(time_slices["test_start"], time_slices["test_end"])
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Load and normalize each variable 
        self.data = []
        filtered_times = None 
        for path, lev in zip(data_paths, levels):
            # Open dataset for this variable
            ds = xr.open_mfdataset(path, combine="by_coords")
            ds = ds.sel(time=time_range)

            if sample_rate == "daily":
                print(f"Set to daily sampling for {path} in the {split} split...")
                ds = ds.resample(time="1D").mean()
            elif sample_rate == "6-hourly":
                print(f"Set to 6-hourly sampling for {path} in the {split} split...")
                ds = ds.resample(time="6h").nearest(tolerance="1h")
            elif sample_rate == "hourly":
                print(f"Set to hourly sampling for {path} in the {split} split...")

            # Compute global min/max from fixed training period
            global_ds = xr.open_mfdataset(path, combine="by_coords")
            global_ds = global_ds.sel(time=slice("2010", "2018"))
            max_val = global_ds.max()[lev].values
            min_val = global_ds.min()[lev].values

            self.norm_stats[lev] = (float(min_val), float(max_val))

            # Normalize to [0, 1]
            arr = (ds[lev] - min_val) / (max_val - min_val)
            arr = arr.load().values  # shape: (time, lat, lon)

            # Drop leap days if enabled
            times = ds["time"].values
            if drop_leap:
                arr, times = self._drop_leap_days(arr, times)
                # Store the filtered times only once (they are the same for all variables)
                if filtered_times is None:
                    filtered_times = times
            else:
                if filtered_times is None:
                    filtered_times = times

            # Add channel dimension (time, 1, H, W)
            arr = torch.from_numpy(arr).float().unsqueeze(1)
            self.data.append(arr)

        # Stack variables into a single tensor (time, channels, H, W)
        self.data = torch.cat(self.data, dim=1)

        # Cache grid size
        _, _, self.H, self.W = self.data.shape

        # before adding temporal encodings
        self.num_var_channels = len(self.levels)  # exclude sin/cos of season
        self.level_to_idx = {lev: i for i, lev in enumerate(self.levels)}

        # Add temporal (seasonal) encodings
        temporal_features, _ = self.temporal_encoding(filtered_times, self.H, self.W)
        self.data = torch.cat([self.data, temporal_features], dim=1)

        # Cache channel size
        _, self.C, _, _ = self.data.shape

    def _drop_leap_days(self, arr, times):
        """
        Remove all leap-day entries (February 29) from the dataset.

        Args:
            arr (np.ndarray): Data array of shape (time, H, W).
            times (np.ndarray): Corresponding datetime64 array.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Filtered data array with leap days removed.
                - Matching filtered time array.
        """
        mask = np.array([
            not (pd.Timestamp(t).month == 2 and pd.Timestamp(t).day == 29)
            for t in times
        ])

        return arr[mask], times[mask]

    def temporal_encoding(self, times, H, W):
        """
        Create cyclical temporal encodings to represent the time of year.
        The encoding adds two channels — sin(day_of_year) and cos(day_of_year).

        Args:
            times (np.ndarray or list of datetime64]): Time steps matching the data array.
            H (int): Spatial grid height.
            W (int): Spatial grid width.

        Returns:
            tuple[torch.Tensor, np.ndarray]:
                - Temporal encoding tensor of shape (time, 2, H, W),
                where channels are [sin_doy, cos_doy].
                - The corresponding (possibly filtered) time array.
        """

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

    def __len__(self):
        """
        Number of available samples.

        Each sample consists of `input_length` past timesteps
        followed by the target at `forecast_horizon`.
        """
        return self.data.shape[0] - self.input_length - self.forecast_horizon

    def __getitem__(self, idx):
        """
        Fetch one training/validation/test sample.

        Args:
            idx (int): Starting index for the input sequence.

        Returns:
            torch_geometric.data.Data: Spatio-temporal graph with:
                - x: Node features for input sequence
                - edge_index: Spatial + temporal edges
                - y: Node labels at forecast horizon
        """
        start = idx
        end = idx + self.input_length
        target_idx = end + self.forecast_horizon - 1

        X = self.data[start:end]       # (input_length, channels, H, W)
        # added: ensure our target doesn't include temporal encodings
        y_full = self.data[target_idx]  # (C_vars + 2, H, W)
        y = y_full[: self.num_var_channels]  # (C_vars, H, W)

        if self.graph_mode:
            graph = to_spatio_temporal_graph(
                X, y, self.H, self.W,
                neighborhood=self.neighborhood,
                graph_type=self.graph_type
            )
            return graph
        else:
            # Return plain tensors for CNN baseline model
            return X, y
        
    @property
    def in_channels(self):
        """Number of input feature channels per node (including seasonality)."""
        return self.C 

    @property
    def out_channels(self):
        """Number of output feature channels per node."""
        # return self.C      # we assume predicting the same variables as input (5 features).
        return self.num_var_channels
    @property
    def normalization_stats(self):
        """Return dictionary of {variable_name: (min, max)} for denormalization."""
        return self.norm_stats
