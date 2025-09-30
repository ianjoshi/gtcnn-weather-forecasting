import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

from data.dataloader import get_dataloaders


def main():

    # Load config
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "utils" / "default_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Config loaded!")

    # Download dataset if not already present
    local_dir = Path(config["data"]["local_dir"])
    if not local_dir.exists() or not any(local_dir.iterdir()):
        print(f"Dataset not found in {local_dir}, downloading...")
        snapshot_download(
            repo_id=config["data"]["repo_id"], 
            repo_type="dataset", 
            local_dir=local_dir,  
            allow_patterns="*"
        )
    else:
        print(f"Dataset already exists at {local_dir}, skipping download.")

    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Try first batch
    first_batch = next(iter(train_loader))
    first_batch = first_batch.to(device)
    print(first_batch)
    print("x:", first_batch.x.shape, first_batch.x.device)
    print("edge_index:", first_batch.edge_index.shape, first_batch.edge_index.device)
    print("y:", first_batch.y.shape, first_batch.y.device)


if __name__ == "__main__":
    main()
