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
        default="./checkpoints/best_gtcnn.pt",
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
    category = "graph_based" if args.model_type in model_config.get("graph_based", {}).keys() else "grid_based"
    model_config = model_config[category]

    # Dataloaders
    outs = get_dataloaders(config=config, model_category=category, eval_mode=True)

    # accept either a single test loader or a tuple of (train,val,test)
    if isinstance(outs, tuple):
        test_loader = outs[-1]
    else:
        test_loader = outs

    # handles Subset wrapper
    base_ds = getattr(test_loader.dataset, "dataset", test_loader.dataset)
    C_in = base_ds.in_channels
    C_out = base_ds.out_channels

    # enables mac
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)

    # Model selection
    model = initialize_model(model_config=model_config, model_type=args.model_type, C_in=C_in, C_out=C_out)
    model = model.to(device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)

    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt.get('epoch', 'N/A')})")

    # Evaluate
    print("Evaluating on test set...")
    metrics = evaluate(model, args.model_type, test_loader, device, config)

    # Save to text file
    report_dir = root_dir / "reports"
    save_report(metrics, args.model_type, ckpt_path, report_dir)


if __name__ == "__main__":
    main()
