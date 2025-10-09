import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml

def download_era5_data(config_path="utils/base_config.yaml"):
    """Download ERA5 dataset from Hugging Face Hub"""
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    local_dir = Path(config["data"]["local_dir"])
    
    # Check if data already exists
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"✅ Dataset already exists at {local_dir}")
        return True
    
    # Download dataset
    print(f"📥 Downloading ERA5 dataset to {local_dir}...")
    try:
        snapshot_download(
            repo_id=config["data"]["repo_id"],
            repo_type="dataset", 
            local_dir=local_dir,
            allow_patterns="*"
        )
        print(f"✅ Download complete!")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 dataset")
    parser.add_argument("--config", default="utils/base_config.yaml")
    args = parser.parse_args()
    
    success = download_era5_data(args.config)
    exit(0 if success else 1)