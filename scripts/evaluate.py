import argparse
import torch
import torch.nn.functional as F
import torch_geometric
import yaml
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from data.dataloader import get_dataloaders
from models.gtcnn import GTCNN
from models.sign import SIGN
from models.cnn3d import CNN3D, SimpleCNN3D  
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test data")
    parser.add_argument(
        "--model_type",
        type=str,
        default="gtcnn",
        choices=["gtcnn", "sign", "cnn3d", "persistence", "climatology"],
        help="Model type to evaluate or 'persistence' for baseline",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None, # For example, for GTCNN: "./checkpoints/best_gtcnn.pt"
        help="Path to checkpoint file (.pt). Optional for persistence. If not provided for a model, loads best_<model>.pt from checkpoints/",
    )
    parser.add_argument(
        "--model_category",
        type=str,
        default="graph_based",
        choices=["graph_based", "grid_based"],
        help="Dataset format (graph-based or grid-based)",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Custom name for the report file (default: report_{model_type}.txt)",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, model_type, loader, device, config):
    if model_type in ["gtcnn", "sign", "cnn3d"]:
        model.eval()
    
    total_loss = 0.0

    all_preds = []
    all_targets = []

    # Get dataset properties
    dataset = loader.dataset
    H, W = dataset.H, dataset.W
    C_out = dataset.out_channels
    T = config["graph"]["input_length"] if "graph" in config else config.get("input_length", 1)
    N_single = H * W  # Single-graph node count (not batched)

    start_time = time.time()
    for batch in tqdm(loader, desc="Evaluating"):
        if model_type == "persistence":
            # Determine data format
            if isinstance(batch, torch_geometric.data.Batch):
                batch = batch.to(device)
                N = H * W
                bs = batch.y.size(0) // N
                y_hat = batch.x.view(bs, T, N, C_out)[:, -1, :, :].reshape(bs * N, C_out)
                y_true = batch.y
            else:  # Grid-based
                X, y_true = batch
                X, y_true = X.to(device), y_true.to(device)
                y_hat = X[:, -1, :, :, :] if X.dim() == 5 and X.shape[1] == T else X[:, :, -1, :, :]

        elif model_type == "climatology":
            # Determine data format (same as persistence)
            if isinstance(batch, torch_geometric.data.Batch):
                batch = batch.to(device)
                N = H * W
                bs = batch.doy.size(0)
                C = dataset.C
                
                climatology = dataset.climatology.to(device)
                clim_batch = climatology[batch.doy - 1]
                y_hat = clim_batch.permute(0, 2, 3, 1).reshape(bs * N, C)
                y_true = batch.y
            else:  # Grid-based
                raise NotImplementedError("Climatology baseline not implemented for grid-based data.")

        elif model_type == "gtcnn":
            batch = batch.to(device)
            N = batch.y.size(0)  # Batched node count (works for GTCNN)

            y_hat = model(batch.x, batch.edge_index, N, T)
            y_true = batch.y

        elif model_type == "sign":
            batch = batch.to(device)

            y_hat = model(batch.x, batch.edge_index, N_single, T)
            y_true = batch.y

        elif model_type == "cnn3d":
            X, y_true = batch
            X, y_true = X.to(device), y_true.to(device)
            y_hat = model(X)
            
            # For 3D CNN, ensure output has the right shape
            # Model should output [B, C_out, H, W], target is [B, C_out, H, W]
            if y_hat.dim() == 4 and y_true.dim() == 4:
                # Flatten spatial dimensions for metric calculation
                y_hat = y_hat.reshape(y_hat.size(0), y_hat.size(1), -1)
                y_true = y_true.reshape(y_true.size(0), y_true.size(1), -1)

        loss = F.mse_loss(y_hat, y_true)
        total_loss += loss.item()

        # Move to CPU and flatten for metrics
        all_preds.append(y_hat.detach().cpu().numpy().ravel())
        all_targets.append(y_true.detach().cpu().numpy().ravel())

    end_time = time.time()
    print(f"Evaluation time: {end_time - start_time:.2f} seconds")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mse = np.mean((all_preds - all_targets) ** 2)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Loss": total_loss / len(loader),
        "Evaluation time": end_time - start_time,
    }

def save_report(metrics, model_type, ckpt_path, report_dir, save_name=None):
    # Save metrics to a text report 
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Use custom save_name if provided, otherwise default to model_type
    if save_name:
        report_path = report_dir / f"report_{save_name}.txt"
    else:
        report_path = report_dir / f"report_{model_type}.txt"

    with open(report_path, "w") as f:
        f.write(f"Performance Report for {model_type.upper()}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:>10}: {v:.6f}\n")
        f.write("=" * 50 + "\n")

    print(f"Report saved to: {report_path}")
    return report_path


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

    elif model_type == "cnn3d": 
        hidden_ch = model_config[model_type]["hidden_channels"]
        num_layers = model_config[model_type].get("num_layers", 4)
        dropout = model_config[model_type].get("dropout", 0.1)
        use_bn = model_config[model_type].get("use_bn", True)
        
        # Choose which version to use
        model_type_variant = model_config[model_type].get("variant", "simple")
        
        if model_type_variant == "full":
            model = CNN3D(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out,
                         num_layers=num_layers, dropout=dropout, use_bn=use_bn)
        else:
            model = SimpleCNN3D(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out,
                               num_layers=num_layers, dropout=dropout, use_bn=use_bn)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    args = parse_args()

    # Load config
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "utils" / "base_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config_path = root_dir / "utils" / "model_config.yaml"
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    print("Config loaded!")

    # Model category
    category = args.model_category
    model_config = model_config[category]

    # Dataloaders - use eval_mode=True for test set
    test_loader = get_dataloaders(config=config, model_category=category, eval_mode=True)
    C_in = test_loader.dataset.in_channels
    C_out = test_loader.dataset.out_channels 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model selection
    ckpt_path = args.model_type # For baselines, this is just the model name
    if args.model_type in ["gtcnn", "sign", "cnn3d"]:
        model = initialize_model(model_config=model_config, model_type=args.model_type, C_in=C_in, C_out=C_out)  
        model = model.to(device)

        # Load checkpoint
        ckpt_path = Path(args.checkpoint) # This overwrites the string with a Path object

        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt.get('epoch', 'N/A')})")

        # Precompute adjacency matrices for SIGN (must be done AFTER loading checkpoint)
        if args.model_type == "sign":
            # Get a SINGLE sample (not batched) to extract edge_index structure
            single_sample = test_loader.dataset[0]
            N = single_sample.y.size(0)  # Number of spatial nodes (H*W)
            T = config["graph"]["input_length"]
            
            # Move edge_index to device and precompute
            edge_index = single_sample.edge_index.to(device)
            model.precompute_adjacency_powers(edge_index, N, T)
            print(f"SIGN precomputation complete: {N} spatial nodes × {T} timesteps = {N*T} nodes per graph")

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate(model, args.model_type, test_loader, device, config)

    # Print results
    print("\n" + "="*50)
    print(f"Evaluation Results for {args.model_type.upper()}:")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric:>10}: {value:.6f}")
    print("="*50)

    # Save detailed report
    report_dir = root_dir / "reports"
    report_path = save_report(metrics, args.model_type, ckpt_path, report_dir, save_name=args.save_name)


if __name__ == "__main__":
    main()
