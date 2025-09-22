import xarray as xr
from pathlib import Path

def load_full_dataset(url: str, consolidated: bool = False):
    """
    Open the full ERA5 Zarr dataset from Google Cloud.
    
    Args:
        url (str): path to the Zarr dataset
        consolidated (bool): whether to use consolidated metadata
    """
    print(f"Opening dataset from: {url}")
    ds = xr.open_zarr(url, consolidated=consolidated)
    print("Full dataset loaded!")
    return ds


def reduce_dataset(ds, cfg):
    """
    Apply preprocessing and reduction to the full dataset based on config.
    Saves train/val/test splits locally as Zarr files.
    On later runs, loads directly from local cache instead of Google Cloud.
    """
    out_dir = Path("../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    train_path = out_dir / "era5_train.zarr"
    val_path   = out_dir / "era5_val.zarr"
    test_path  = out_dir / "era5_test.zarr"

    # If all splits already exist, just load them
    if train_path.exists() and val_path.exists() and test_path.exists():
        print("Using cached local splits from data/processed/")
        return {
            "train": xr.open_zarr(train_path, consolidated=False),
            "val":   xr.open_zarr(val_path, consolidated=False),
            "test":  xr.open_zarr(test_path, consolidated=False),
        }

    print("Reducing dataset from source (this may take a while)...")

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # 1. Variable selection
    reduced_ds = ds[data_cfg["variables_to_keep"]]

    # 2. Spatial region
    reduced_ds = reduced_ds.sel(
        latitude=slice(data_cfg["region"]["lat_max"], data_cfg["region"]["lat_min"]),
        longitude=slice(data_cfg["region"]["lon_min"], data_cfg["region"]["lon_max"])
    )

    # 3. Temporal range (whole span from train start to test end)
    reduced_ds = reduced_ds.sel(
        time=slice(data_cfg["time"]["train_start"], data_cfg["time"]["test_end"])
    )

    # 4. Levels for multi-level variables
    if "level" in reduced_ds.dims and data_cfg.get("levels"):
        reduced_ds = reduced_ds.sel(level=data_cfg["levels"])

    # # 5. Optional time subsampling
    # if data_cfg.get("time_subsample"):
    #     reduced_ds = reduced_ds.isel(time=slice(0, None, data_cfg["time_subsample"]))

    # Split into train/val/test and save locally
    splits = {
        "train": (data_cfg["time"]["train_start"], data_cfg["time"]["train_end"], train_path),
        "val":   (data_cfg["time"]["val_start"],   data_cfg["time"]["val_end"],   val_path),
        "test":  (data_cfg["time"]["test_start"],  data_cfg["time"]["test_end"],  test_path),
    }

    result = {}
    for split, (start, end, path) in splits.items():
        print(f"Creating {split} set ({start} → {end})...")
        split_ds = reduced_ds.sel(time=slice(start, end))
        split_ds = split_ds.chunk({"time": train_cfg["input_length"], "latitude": 50, "longitude": 50})
        split_ds.to_zarr(path, mode="w", safe_chunks=False)
        result[split] = split_ds
        print(f"Saved {split} set to {path}")

    return result
