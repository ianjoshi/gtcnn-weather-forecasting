import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import yaml
from pathlib import Path
from huggingface_hub import snapshot_download
import time

from data.dataloader import get_dataloaders
from models.gtcnn import GTCNN
from models.sign import SIGN
from models.cnn3d import CNN3D, SimpleCNN3D

def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models on ERA5 data")
    parser.add_argument(
        "--model_type",
        type=str,
        default="gtcnn",
        choices=["gtcnn", "sign", "cnn3d"],
        help="Model type to train",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Custom name for checkpoint and summary files (default: {model_type})",
    )
    return parser.parse_args()


def train_one_epoch(model, model_type, loader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    # Get single-graph node count for SIGN (H*W, not batched)
    if model_type == "sign":
        H, W = loader.dataset.H, loader.dataset.W
        N_single = H * W

    for batch in progress_bar:
        optimizer.zero_grad()

        if model_type == "gtcnn": 
            batch = batch.to(device)
            N = batch.y.size(0)  # number of spatial nodes (H*W) - batched
            T = config["graph"]["input_length"]

            y_hat = model(batch.x, batch.edge_index, N, T)
            loss = F.mse_loss(y_hat, batch.y)

        elif model_type == "sign":
            batch = batch.to(device)
            T = config["graph"]["input_length"]

            y_hat = model(batch.x, batch.edge_index, N_single, T)
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
    
    # Get single-graph node count for SIGN (H*W, not batched)
    if model_type == "sign":
        H, W = loader.dataset.H, loader.dataset.W
        N_single = H * W
    
    with torch.no_grad():
        for batch in progress_bar:
            if model_type == "gtcnn": 
                batch = batch.to(device)
                N = batch.y.size(0)
                T = config["graph"]["input_length"]

                y_hat = model(batch.x, batch.edge_index, N, T)
                loss = F.mse_loss(y_hat, batch.y)

            elif model_type == "sign":
                batch = batch.to(device)
                T = config["graph"]["input_length"]

                y_hat = model(batch.x, batch.edge_index, N_single, T)
                loss = F.mse_loss(y_hat, batch.y)

            elif model_type == "cnn3d":
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = F.mse_loss(y_hat, y)

            total_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": loss.item()})

    return total_loss / len(loader)

def initialize_model(model_config, model_type, C_in, C_out):

    if model_type == "gtcnn": 

        hidden_ch = model_config[model_type]["hidden_channels"] 
        K = model_config[model_type]["chebyshev_order"] 
        num_layers = model_config[model_type]["num_layers"] 
        dropout = model_config[model_type]["dropout"] 
        model = GTCNN(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out, K=K, 
                      num_layers=num_layers, dropout=dropout)
                      
    elif model_type == "sign":
        hidden_ch = model_config[model_type]["hidden_channels"]
        K = model_config[model_type]["K"]
        dropout = model_config[model_type]["dropout"]
        use_bn = model_config[model_type]["use_bn"]
        
        model = SIGN(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out,
                    K=K, dropout=dropout, use_bn=use_bn)
                      
    # Choose which version to use - CNN3D for full U-Net style, SimpleCNN3D for faster training
        
    elif model_type == "cnn3d": 
        hidden_ch = model_config[model_type]["hidden_channels"]
        num_layers = model_config[model_type].get("num_layers", 4)
        dropout = model_config[model_type].get("dropout", 0.1)
        use_bn = model_config[model_type].get("use_bn", True)
        
        # Choose which version to use - CNN3D for full U-Net style, SimpleCNN3D for faster training
        model_type_variant = model_config[model_type].get("variant", "simple")
        
        if model_type_variant == "full":
            model = CNN3D(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out,
                         num_layers=num_layers, dropout=dropout, use_bn=use_bn)
        else:
            model = SimpleCNN3D(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out,
                               num_layers=num_layers, dropout=dropout, use_bn=use_bn)

    else:
        raise ValueError(f"Unknown model type !!!")
    
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

    # Model selection
    model = initialize_model(model_config=model_config, model_type=args.model_type, C_in=C_in, C_out=C_out)  
    model = model.to(device)
    
    # Precompute adjacency matrices for SIGN
    if args.model_type == "sign":
        # Get a SINGLE sample (not batched) to extract edge_index structure
        single_sample = train_loader.dataset[0]  # Get one graph directly from dataset
        N = single_sample.y.size(0)  # Number of spatial nodes (H*W)
        T = config["graph"]["input_length"]
        
        # Move edge_index to device and precompute
        edge_index = single_sample.edge_index.to(device)
        model.precompute_adjacency_powers(edge_index, N, T)
        print(f"SIGN precomputation complete: {N} spatial nodes × {T} timesteps = {N*T} nodes per graph")
    
    optimizer = optim.AdamW(model.parameters(),
                            lr=float(config["training"]["learning_rate"]),
                            weight_decay=float(config["training"]["weight_decay"]))

    # Prepare checkpoint directory
    ckpt_dir = root_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Prepare training summary directory
    summary_dir = root_dir / "training_summaries"
    summary_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    best_state = None
    patience = config["training"].get("early_stopping", 5)
    epochs_no_improve = 0

    # Start training timer
    training_start_time = time.time()

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

    # Calculate total training time
    total_training_time = time.time() - training_start_time
    
    # Save best model at the end
    if best_state is not None:
        # Use custom save_name if provided, otherwise default to model_type
        save_name = args.save_name if args.save_name else args.model_type
        
        ckpt_path = ckpt_dir / f"{save_name}.pt"
        torch.save(best_state, ckpt_path)
        print(f"\nTraining complete! Best model saved to {ckpt_path} (val_loss={best_val_loss:.4f})")
        print(f"Total training time: {total_training_time/60:.2f} minutes ({total_training_time:.2f}s)")

        # Save training summary
        summary_path = summary_dir / f"training_summary_{save_name}.txt"
        with open(summary_path, "w") as f:
            f.write(f"Training Summary for {args.model_type.upper()}\n")
            f.write(f"Best model saved to: {ckpt_path}\n")
            f.write(f"Best validation loss: {best_val_loss:.4f}\n")
            f.write(f"Total training time: {total_training_time/60:.2f} minutes ({total_training_time:.2f}s)\n")
            f.write(f"Total epochs trained: {epoch}\n")

    else:
        print("\nNo model was saved (training may have failed).")



if __name__ == "__main__":
    main()
