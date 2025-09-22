import xarray as xr

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

    Args:
        ds (xarray.Dataset): full ERA5 dataset
        cfg (dict): config["data"] dictionary
    """
    print("Reducing dataset...")

    # 1. Variable selection
    variables_to_keep = cfg["variables_to_keep"]
    reduced_ds = ds[variables_to_keep]

    # 2. Spatial region
    reduced_ds = reduced_ds.sel(
        latitude=slice(cfg["region"]["lat_max"], cfg["region"]["lat_min"]),
        longitude=slice(cfg["region"]["lon_min"], cfg["region"]["lon_max"])
    )

    # 3. Temporal range (whole span from train start to test end)
    reduced_ds = reduced_ds.sel(
        time=slice(cfg["time"]["train_start"], cfg["time"]["test_end"])
    )

    # 4. Levels for multi-level variables
    if "level" in reduced_ds.dims and cfg.get("levels"):
        reduced_ds = reduced_ds.sel(level=cfg["levels"])

    print("Reduced dataset ready!")
    return reduced_ds
