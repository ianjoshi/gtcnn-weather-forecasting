import argparse
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score

from data.dataloader import get_dataloaders
from models.gtcnn import GTCNN
from models.cnn3d import CNN3D, SimpleCNN3D  # ADD THIS IMPORT


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test data")
    parser.add_argument(
        "--model_type",
        type=str,
        default="gtcnn",
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,  # CHANGE: Make default None to auto-detect
        help="Path to checkpoint file (.pt). If not provided, loads best_<model>.pt from checkpoints/",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, model_type, loader, device, config):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    for batch in loader:
        if model_type == "gtcnn":
            batch = batch.to(device)
            N = batch.y.size(0)
            T = config["graph"]["input_length"]

            y_hat = model(batch.x, batch.edge_index, N, T)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"report_{model_type}_{timestamp}.txt"

    with open(report_path, "w") as f:
        f.write(f"Performance Report for {model_type.upper()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    category = "graph_based" if args.model_type in model_config.get("graph_based", {}).keys() else "grid_based"
    model_config = model_config[category]

    # Dataloaders - use eval_mode=True for test set
    test_loader = get_dataloaders(config=config, model_category=category, eval_mode=True)
    C_in = test_loader.dataset.in_channels
    C_out = test_loader.dataset.out_channels 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model initialization
    model = initialize_model(model_config=model_config, model_type=args.model_type, C_in=C_in, C_out=C_out)  
    model = model.to(device)

    # Load checkpoint - auto-detect if not provided
    if args.checkpoint is None:
        ckpt_path = root_dir / "checkpoints" / f"best_{args.model_type}.pt"
    else:
        ckpt_path = Path(args.checkpoint)
    
    print(f"Looking for checkpoint at: {ckpt_path}")

    if not ckpt_path.exists():
        # Try to find any checkpoint with the model type in the name
        ckpt_dir = root_dir / "checkpoints"
        matching_ckpts = list(ckpt_dir.glob(f"*{args.model_type}*.pt"))
        if matching_ckpts:
            ckpt_path = matching_ckpts[0]
            print(f"Found alternative checkpoint: {ckpt_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found for model type '{args.model_type}'. "
                                  f"Expected: {ckpt_path} or any file containing '{args.model_type}' in checkpoints/")

    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch_info = ckpt.get('epoch', 'N/A')
    else:
        # Direct state dict
        model.load_state_dict(ckpt)
        epoch_info = "N/A"
        
    print(f"Loaded checkpoint from {ckpt_path} (epoch {epoch_info})")

    # Print model info
    print(f"Model: {args.model_type}")
    print(f"Input channels: {C_in}, Output channels: {C_out}")
    print(f"Test samples: {len(test_loader.dataset)}")

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
    report_path = save_report(metrics, args.model_type, ckpt_path, report_dir)
    
    # Also save metrics in a more machine-readable format
    metrics_path = report_dir / f"metrics_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump({args.model_type: {k: float(v) for k, v in metrics.items()}}, f)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
