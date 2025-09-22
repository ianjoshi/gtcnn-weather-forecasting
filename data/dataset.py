import torch
from torch.utils.data import Dataset

class ERA5Dataset(Dataset):
    def __init__(self, xr_dataset, input_vars, target_var,
                 input_length=7, forecast_horizon=1,
                 split="train"):
        """
        ERA5 dataset for Graph ML.

        Args:
            xr_dataset (xarray.Dataset): preprocessed ERA5 dataset
            input_vars (list): list of predictor variable names
            target_var (str): target variable (e.g. "2m_temperature")
            input_length (int): number of past timesteps to use as input
            forecast_horizon (int): how far ahead to predict (days)
            split (str): "train", "val", or "test"
        """
        self.ds = xr_dataset
        self.input_vars = input_vars
        self.target_var = target_var
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon

        # Time-based splits
        if split == "train":
            self.subset = self.ds.sel(time=slice("2010-01-01", "2017-12-31"))
        elif split == "val":
            self.subset = self.ds.sel(time=slice("2018-01-01", "2019-12-31"))
        elif split == "test":
            self.subset = self.ds.sel(time=slice("2020-01-01", "2022-12-31"))
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.times = self.subset.time.values

    def __len__(self):
        return len(self.times) - self.input_length - self.forecast_horizon

    def __getitem__(self, idx):
        start = idx
        end = idx + self.input_length
        input_slice = self.subset.isel(time=slice(start, end))
        target_idx = end + self.forecast_horizon - 1
        target_slice = self.subset.isel(time=target_idx)

        X = torch.tensor(input_slice[self.input_vars].to_array().values,
                         dtype=torch.float32)  # (n_vars, input_length, lat, lon)
        y = torch.tensor(target_slice[self.target_var].values,
                         dtype=torch.float32)  # (lat, lon)

        return X, y
