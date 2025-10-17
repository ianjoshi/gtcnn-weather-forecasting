import argparse
import torch
import torch.nn.functional as F
import torch_geometric
import yaml
from pathlib import Path
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score

from data.dataloader import get_dataloaders
from models.gtcnn import GTCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test data")
    parser.add_argument(
        "--model_type",
        type=str,
        default="gtcnn",
        choices=["gtcnn", "cnn3d", "persistence"],
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
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, model_type, loader, device, config):
    if model_type in ["gtcnn", "cnn3d"]:
        model.eval()
    
    total_loss = 0.0

    all_preds = []
    all_targets = []

    # Get dataset properties
    dataset = loader.dataset
    H, W = dataset.H, dataset.W
    C_out = dataset.out_channels
    T = config["graph"]["input_length"] if "graph" in config else config.get("input_length", 1)

    for batch in loader:
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

        elif model_type == "gtcnn":
            batch = batch.to(device)
            N = batch.y.size(0)

            y_hat = model(batch.x, batch.edge_index, N, T)
            y_true = batch.y

        elif model_type == "cnn3d":
            X, y_true = batch
            X, y_true = X.to(device), y_true.to(device)
            y_hat = model(X)

        loss = F.mse_loss(y_hat, y_true)
        total_loss += loss.item()

        # Move to CPU and flatten for metrics
        all_preds.append(y_hat.detach().cpu().numpy().ravel())
        all_targets.append(y_true.detach().cpu().numpy().ravel())

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
    }


def save_report(metrics, model_type, ckpt_path, report_dir):
    # Save metrics to a text report 
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"report_{model_type}.txt"

    with open(report_path, "w") as f:
        f.write(f"Performance Report for {model_type.upper()}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write("-" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:>10}: {v:.6f}\n")
        f.write("-" * 50 + "\n")

    print(f"Report saved to: {report_path}")

def initialize_model(model_config, model_type, C_in, C_out):

    if model_type == "gtcnn": 

        hidden_ch = model_config[model_type]["hidden_channels"] 
        K = model_config[model_type]["chebyshev_order"] 
        num_layers = model_config[model_type]["num_layers"] 
        dropout = model_config[model_type]["dropout"] 
        model = GTCNN(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out, K=K, 
                      num_layers=num_layers, dropout=dropout)

    elif model_type == "cnn3d": 

        hidden_ch = model_config[model_type]["hidden_channels"]
        # model = CNN3D(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out)

    else:
        raise ValueError(f"Unknown model type !!!")
    
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

    # Dataloaders
    test_loader = get_dataloaders(config=config, model_category=category, eval_mode=True)
    C_in = test_loader.dataset.in_channels
    C_out = test_loader.dataset.out_channels 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model selection
    ckpt_path = None
    if args.model_type != "persistence":
        model = initialize_model(model_config=model_config, model_type=args.model_type, C_in=C_in, C_out=C_out)  
        model = model.to(device)

        # Load checkpoint
        ckpt_path = Path(args.checkpoint)

        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt.get('epoch', 'N/A')})")
    else:
        model = None
        print("Evaluating persistence baseline (no model required).")

    # Evaluate
    print("Evaluating on test set...")
    metrics = evaluate(model, args.model_type, test_loader, device, config)

    # Save to text file
    report_dir = root_dir / "reports"
    save_report(metrics, args.model_type, ckpt_path or "persistence", report_dir)


if __name__ == "__main__":
    main()
