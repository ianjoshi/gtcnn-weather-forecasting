import torch
from torch.utils.data import Dataset

class ERA5Dataset(Dataset):
    def __init__(self, xr_dataset, input_vars, target_var, config,
                 input_length=7, forecast_horizon=1,
                 split="train", load_into_ram=False):
        """
        ERA5 dataset for Graph ML.

        Args:
            xr_dataset (xarray.Dataset): preprocessed ERA5 dataset
            input_vars (list): predictor variable names
            target_var (str): target variable (e.g. "2m_temperature")
            config (dict): full config dict, expects config["data"]["time"]
            input_length (int): number of past timesteps to use as input
            forecast_horizon (int): how far ahead to predict (timesteps)
            split (str): "train", "val", or "test"
            load_into_ram (bool): if True, loads the split into memory as numpy arrays
        """
        self.input_vars = input_vars
        self.target_var = target_var
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon

        # Get time ranges from config
        time_cfg = config["time"]

        if split == "train":
            subset = xr_dataset.sel(time=slice(time_cfg["train_start"], time_cfg["train_end"]))
        elif split == "val":
            subset = xr_dataset.sel(time=slice(time_cfg["val_start"], time_cfg["val_end"]))
        elif split == "test":
            subset = xr_dataset.sel(time=slice(time_cfg["test_start"], time_cfg["test_end"]))
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Optionally load entire split into RAM (numpy-backed)
        if load_into_ram:
            print(f"Loading {split} split into RAM ")
            subset = subset.load()

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

        # Already in RAM if load_into_ram=True
        X_np = input_slice[self.input_vars].to_array().to_numpy()
        y_np = target_slice[self.target_var].to_numpy()

        X = torch.from_numpy(X_np).float()  # (n_vars, input_length, lat, lon)
        y = torch.from_numpy(y_np).float()  # (lat, lon)

        return X, y
