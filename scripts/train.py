import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import yaml
from huggingface_hub import snapshot_download

from data.dataloader import get_dataloaders
from models.gtcnn import GTCNN
from models.scalable_gtcnn import ScalableGTCNN

def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models on ERA5 data")
    parser.add_argument(
        "--model_type",
        type=str,
        default="gtcnn",
        help="Model type to train",
    )
    return parser.parse_args()


def train_one_epoch(model, model_type, loader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        if model_type == "gtcnn": 
            batch = batch.to(device)
            N = batch.y.size(0)  # number of spatial nodes (H*W)
            T = config["graph"]["input_length"]

            y_hat = model(batch.x, batch.edge_index, N, T)
            loss = F.mse_loss(y_hat, batch.y)

        elif model_type == "scalable_gtcnn":
            batch = batch.to(device)
            N = batch.y.size(0)  # number of spatial nodes (H*W)
            T = config["graph"]["input_length"]

            y_hat = model(batch.x, batch.edge_index, N, T)
            loss = F.mse_loss(y_hat, batch.y)

        elif model_type == "cnn3d":  
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = F.mse_loss(y_hat, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        progress_bar.set_postfix({"batch_loss": loss.item()})

    return total_loss / len(loader)


def validate(model, model_type, loader, device, config):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            if model_type == "gtcnn": 
                batch = batch.to(device)
                N = batch.y.size(0)
                T = config["graph"]["input_length"]

                y_hat = model(batch.x, batch.edge_index, N, T)
                loss = F.mse_loss(y_hat, batch.y)

            elif model_type == "scalable_gtcnn":
                batch = batch.to(device)
                N = batch.y.size(0)
                T = config["graph"]["input_length"]

                y_hat = model(batch.x, batch.edge_index, N, T)
                loss = F.mse_loss(y_hat, batch.y)

            elif model_type == "cnn3d":
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = F.mse_loss(y_hat, y)

            total_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": loss.item()})

    return total_loss / len(loader)

def initialize_model(model_config, model_type, C_in, C_out, H=None, W=None):
    """
    Initialize model based on type and configuration.
    
    Args:
        model_config: Model configuration dictionary
        model_type: Type of model to initialize
        C_in: Number of input channels
        C_out: Number of output channels
        H, W: Grid dimensions (only required for ScalableGTCNN)
    """
    if model_type == "gtcnn": 
        hidden_ch = model_config[model_type]["hidden_channels"] 
        K = model_config[model_type]["chebyshev_order"] 
        num_layers = model_config[model_type]["num_layers"] 
        dropout = model_config[model_type]["dropout"] 
        model = GTCNN(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out, K=K, 
                      num_layers=num_layers, dropout=dropout)

    elif model_type == "scalable_gtcnn":
        # Create base GTCNN first
        hidden_ch = model_config[model_type]["hidden_channels"] 
        K = model_config[model_type]["chebyshev_order"] 
        num_layers = model_config[model_type]["num_layers"] 
        dropout = model_config[model_type]["dropout"]
        
        base_gtcnn = GTCNN(
            in_channels=C_in, 
            hidden_channels=hidden_ch, 
            out_channels=C_out, 
            K=K, 
            num_layers=num_layers, 
            dropout=dropout
        )
        
        # Wrap with ScalableGTCNN
        if H is None or W is None:
            raise ValueError("H and W must be provided for ScalableGTCNN")
            
        model = ScalableGTCNN(
            gtcnn=base_gtcnn,
            H=H, W=W,
            num_neighbors=model_config[model_type]["num_neighbors"],
            neighborhood=model_config[model_type]["neighborhood"],
            sampling_strategy=model_config[model_type]["sampling_strategy"]
        )

    elif model_type == "cnn3d": 
        hidden_ch = model_config[model_type]["hidden_channels"]
        # model = CNN3D(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def main():
    args = parse_args()

    # Load both configs
    root_dir = Path(__file__).resolve().parent.parent
    base_config_path = root_dir / "utils" / "base_config.yaml"
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config_path = root_dir / "utils" / "model_config.yaml"
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
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

    # Model category
    category = "graph_based" if args.model_type in model_config.get("graph_based", {}).keys() else "grid_based"
    model_config = model_config[category]

    # Dataloaders
    train_loader, val_loader = get_dataloaders(config=config, model_category=category, eval_mode=False)
    C_in = train_loader.dataset.in_channels
    C_out = train_loader.dataset.out_channels 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Get grid dimensions from dataset (only needed for ScalableGTCNN)
    H = train_loader.dataset.H
    W = train_loader.dataset.W
    
    # Model selection
    if args.model_type == "scalable_gtcnn":
        model = initialize_model(
            model_config=model_config, 
            model_type=args.model_type, 
            C_in=C_in, 
            C_out=C_out,
            H=H, W=W
        )
    else:
        model = initialize_model(
            model_config=model_config, 
            model_type=args.model_type, 
            C_in=C_in, 
            C_out=C_out
        )
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=float(config["training"]["learning_rate"]),
                            weight_decay=float(config["training"]["weight_decay"]))

    # Prepare checkpoint directory
    ckpt_dir = root_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    best_state = None
    patience = config["training"].get("early_stopping", 5)
    epochs_no_improve = 0

    # Training loop
    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")

        train_loss = train_one_epoch(model, args.model_type, train_loader, optimizer, device, config)
        val_loss = validate(model, args.model_type, val_loader, device, config)

        print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config
            }
            print(f"New best model found at epoch {epoch}!")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs...")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement in {patience}).")
            break

    # Save best model at the end
    if best_state is not None:
        ckpt_path = ckpt_dir / f"best_{args.model_type}.pt"
        torch.save(best_state, ckpt_path)
        print(f"\nTraining complete! Best model saved to {ckpt_path} (val_loss={best_val_loss:.4f})")
    else:
        print("\nNo model was saved (training may have failed).")



if __name__ == "__main__":
    main()
