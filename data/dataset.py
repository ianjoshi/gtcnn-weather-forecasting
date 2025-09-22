import torch
from torch.utils.data import Dataset
from pathlib import Path
import xarray as xr


class ERA5Dataset(Dataset):
    def __init__(self, split, input_vars, target_var, config, load_into_ram=False):
        """
        ERA5 dataset for Graph ML.

        Args:
            split (str): "train", "val", or "test"
            input_vars (list): predictor variable names
            target_var (str): target variable (e.g. "2m_temperature")
            config (dict): full config dict (expects ["training"] section)
            load_into_ram (bool): if True, load the whole split into RAM
        """
        self.input_vars = input_vars
        self.target_var = target_var
        self.input_length = config["training"]["input_length"]
        self.forecast_horizon = config["training"]["forecast_horizon"]

        # Path to local cached split
        path = Path("data/processed") / f"era5_{split}.zarr"
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path}. Did you run reduce_dataset() first?"
            )

        print(f"Opening {split} split from {path}")
        subset = xr.open_zarr(path, consolidated=False)

        # Optionally load entire split into RAM
        if load_into_ram:
            approx_size_gb = subset.nbytes / 1e9
            print(f"Loading {split} split into RAM "
                  f"(~{approx_size_gb:.2f} GB)...")
            subset = subset.load()
            print(f"{split} split now in memory")

        self.subset = subset
        self.times = self.subset.time.values

    def __len__(self):
        return len(self.times) - self.input_length - self.forecast_horizon

    def __getitem__(self, idx):
        start = idx
        end = idx + self.input_length
        input_slice = self.subset.isel(time=slice(start, end))
        target_idx = end + self.forecast_horizon - 1
        target_slice = self.subset.isel(time=target_idx)

        # Already in memory if load_into_ram=True
        X_np = input_slice[self.input_vars].to_array().to_numpy()
        y_np = target_slice[self.target_var].to_numpy()

        X = torch.from_numpy(X_np).float()  # (n_vars, input_length, lat, lon)
        y = torch.from_numpy(y_np).float()  # (lat, lon)

        return X, y
