import argparse
import random
import numpy as np
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import yaml
from huggingface_hub import snapshot_download
from data.dataloader import get_dataloaders
from models.gtcnn import GTCNN
from models.sign import SIGN
from models.cnn3d import CNN3D, SimpleCNN3D
from utils.losses import kinetic_energy_loss, kinetic_energy_metrics


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
        "--physics_loss",
        action="store_true",
        help="Use physics-based kinetic energy loss (GTCNN only)",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="best_cnn3d",
        help="Custom name for checkpoint and summary files (default: {model_type})",
    )
    parser.add_argument(
        "--assert_shapes",
        action="store_true",
        help="Optional flag to verify tensor shapes on first batch",
    )

    args = parser.parse_args()

    # Enforce that --physics_loss is only valid with GTCNN
    if args.physics_loss and args.model_type != "gtcnn":
        parser.error("--physics_loss can only be used when --model_type is 'gtcnn'.")
        
    return args


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    """Automatically pick best device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ------------------------------------------------------------
# Physics-Aware Loss
# ------------------------------------------------------------
def compute_total_loss(
    y_hat, batch, norm_stats, u_idx, v_idx,
    lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input,
    ref_override=None,
):
    """Combines standard MSE and optional kinetic energy loss."""
    loss_mse = F.mse_loss(y_hat, batch.y)
    loss_total = loss_mse

    if lambda_ke > 0:
        if ref_override is not None:
            ref = ref_override
        elif ke_use_truth:
            ref = batch.y
        else:
            ref = None

        if ref is not None:
            loss_ke = kinetic_energy_loss(
                y_hat, ref, norm_stats, u_idx, v_idx, lambda_ke, mode=ke_mode
            )
            loss_total = loss_total + loss_ke

    return loss_total


# ------------------------------------------------------------
# Training + Validation
# ------------------------------------------------------------
def train_one_epoch(model, model_type, loader, optimizer, device, config,
                    T=None, norm_stats=None, u_idx=None, v_idx=None,
                    lambda_ke=0.0, ke_mode="abs", ke_use_truth=True,
                    ke_compare_to_last_input=False, do_assert=False, use_physics=False):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    first = True

    for batch in progress_bar:
        optimizer.zero_grad()

        if model_type == "gtcnn":
            batch = batch.to(device)
            N = batch.y.size(0)
            C_out = batch.y.size(1)

            if do_assert and first:
                assert batch.x.dim() == 2, f"Expected x [N*T, C_in], got {tuple(batch.x.shape)}"
                assert batch.y.dim() == 2, f"Expected y [N, C_out], got {tuple(batch.y.shape)}"
                first = False

            use_amp = device.type == "cuda"
            if use_amp:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    y_hat = model(batch.x, batch.edge_index, N, T)
                    if use_physics and lambda_ke > 0:
                        ref_override = None
                        if ke_compare_to_last_input:
                            start = (T - 1) * N
                            end = T * N
                            ref_override = batch.x[start:end, :C_out]
                        loss_total = compute_total_loss(
                            y_hat, batch, norm_stats, u_idx, v_idx,
                            lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input,
                            ref_override=ref_override,
                        )
                    else:
                        loss_total = F.mse_loss(y_hat, batch.y)
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                y_hat = model(batch.x, batch.edge_index, N, T)
                if use_physics and lambda_ke > 0:
                    ref_override = None
                    if ke_compare_to_last_input:
                        start = (T - 1) * N
                        end = T * N
                        ref_override = batch.x[start:end, :C_out]
                    loss_total = compute_total_loss(
                        y_hat, batch, norm_stats, u_idx, v_idx,
                        lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input,
                        ref_override=ref_override,
                    )
                else:
                    loss_total = F.mse_loss(y_hat, batch.y)
                loss_total.backward()
                optimizer.step()

        elif model_type == "sign":
            batch = batch.to(device)
            H, W = loader.dataset.H, loader.dataset.W
            N_single = H * W
            T = config["graph"]["input_length"]
            y_hat = model(batch.x, batch.edge_index, N_single, T)
            loss_total = F.mse_loss(y_hat, batch.y)
            loss_total.backward()
            optimizer.step()

        elif model_type == "cnn3d":
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss_total = F.mse_loss(y_hat, y)
            loss_total.backward()
            optimizer.step()

        total_loss += loss_total.item()
        progress_bar.set_postfix({"batch_loss": loss_total.item()})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, model_type, loader, device, config,
             T=None, norm_stats=None, u_idx=None, v_idx=None,
             lambda_ke=0.0, ke_mode="abs", ke_use_truth=True,
             ke_compare_to_last_input=False, use_physics=False):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Validating", leave=False)
    sum_ke_pointwise, sum_ke_global = 0.0, 0.0

    for batch in progress_bar:
        if model_type == "gtcnn":
            batch = batch.to(device)
            N = batch.y.size(0)
            C_out = batch.y.size(1)

            y_hat = model(batch.x, batch.edge_index, N, T)
            if use_physics and lambda_ke > 0:
                ref_override = None
                if ke_compare_to_last_input:
                    start = (T - 1) * N
                    end = T * N
                    ref_override = batch.x[start:end, :C_out]
                loss_total = compute_total_loss(
                    y_hat, batch, norm_stats, u_idx, v_idx,
                    lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input,
                    ref_override=ref_override,
                )
                ke_metrics = kinetic_energy_metrics(y_hat, batch.y, norm_stats, u_idx, v_idx)
                sum_ke_pointwise += ke_metrics["ke_pointwise_mse"].item()
                sum_ke_global += ke_metrics["ke_global_mse"].item()
            else:
                loss_total = F.mse_loss(y_hat, batch.y)

        elif model_type == "sign":
            batch = batch.to(device)
            H, W = loader.dataset.H, loader.dataset.W
            N_single = H * W
            T = config["graph"]["input_length"]
            y_hat = model(batch.x, batch.edge_index, N_single, T)
            loss_total = F.mse_loss(y_hat, batch.y)

        elif model_type == "cnn3d":
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss_total = F.mse_loss(y_hat, y)

        total_loss += loss_total.item()
        progress_bar.set_postfix({"val_loss": f"{loss_total.item():.4f}"})

    if use_physics and model_type == "gtcnn" and lambda_ke > 0:
        avg_ke_pointwise = sum_ke_pointwise / max(1, len(loader))
        avg_ke_global = sum_ke_global / max(1, len(loader))
        print(f"Validation KE | pointwise: {avg_ke_pointwise:.3e} | global: {avg_ke_global:.3e}")

    return total_loss / len(loader)


# ------------------------------------------------------------
# Model Initialization
# ------------------------------------------------------------
def initialize_model(model_config, model_type, C_in, C_out):
    if model_type == "gtcnn":
        cfg = model_config[model_type]
        model = GTCNN(
            in_channels=C_in,
            hidden_channels=cfg["hidden_channels"],
            out_channels=C_out,
            K=cfg["chebyshev_order"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    elif model_type == "sign":
        cfg = model_config[model_type]
        model = SIGN(
            in_channels=C_in,
            hidden_channels=cfg["hidden_channels"],
            out_channels=C_out,
            K=cfg["K"],
            dropout=cfg["dropout"],
            use_bn=cfg["use_bn"],
        )
    elif model_type == "cnn3d":
        cfg = model_config[model_type]
        variant = cfg.get("variant", "simple")
        if variant == "full":
            model = CNN3D(
                in_channels=C_in,
                hidden_channels=cfg["hidden_channels"],
                out_channels=C_out,
                num_layers=cfg.get("num_layers", 4),
                dropout=cfg.get("dropout", 0.1),
                use_bn=cfg.get("use_bn", True),
            )
        else:
            model = SimpleCNN3D(
                in_channels=C_in,
                hidden_channels=cfg["hidden_channels"],
                out_channels=C_out,
                num_layers=cfg.get("num_layers", 3),
                dropout=cfg.get("dropout", 0.1),
                use_bn=cfg.get("use_bn", True),
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def main():
    args = parse_args()
    seed_everything(42)

    # Load configs
    root_dir = Path(__file__).resolve().parent.parent
    with open(root_dir / "utils" / "base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(root_dir / "utils" / "model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    print("Config loaded!")

    # Dataset handling
    local_dir = Path(config["data"]["local_dir"])
    if not local_dir.exists() or not any(local_dir.iterdir()):
        print(f"Dataset not found in {local_dir}, downloading...")
        snapshot_download(
            repo_id=config["data"]["repo_id"],
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns="*",
        )
    else:
        print(f"Dataset already exists at {local_dir}, skipping download.")

    # Model category
    category = "graph_based" if args.model_type in model_config.get("graph_based", {}).keys() else "grid_based"
    model_config = model_config[category]

    # Dataloaders
    train_loader, val_loader = get_dataloaders(config=config, model_category=category, eval_mode=False)
    base_ds = getattr(train_loader.dataset, "dataset", train_loader.dataset)
    C_in, C_out = base_ds.in_channels, base_ds.out_channels

    # KE-related values (for physics loss)
    norm_stats = getattr(base_ds, "normalization_stats", None)
    u_idx, v_idx = None, None
    if args.model_type == "gtcnn" and args.physics_loss:
        level_to_idx = base_ds.level_to_idx
        u_idx = level_to_idx[config["data"]["wind_u_key"]]
        v_idx = level_to_idx[config["data"]["wind_v_key"]]
        print("u_idx,v_idx =", u_idx, v_idx)

    lambda_ke = float(config["training"].get("lambda_ke", 0.0))
    ke_mode = config["training"].get("ke_mode", "abs")
    ke_use_truth = config["training"].get("ke_use_truth", True)
    ke_compare_to_last_input = config["training"].get("ke_compare_to_last_input", False)
    T = int(config["graph"]["input_length"])

    # Device + Model
    device = pick_device()
    print("Using device:", device)
    model = initialize_model(model_config=model_config, model_type=args.model_type, C_in=C_in, C_out=C_out)
    model = model.to(device)

    # SIGN precomputation
    if args.model_type == "sign":
        single_sample = train_loader.dataset[0]
        N = single_sample.y.size(0)
        T_sign = config["graph"]["input_length"]
        edge_index = single_sample.edge_index.to(device)
        model.precompute_adjacency_powers(edge_index, N, T_sign)
        print(f"SIGN precomputation complete: {N} nodes × {T_sign} timesteps")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    ckpt_dir = root_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    summary_dir = root_dir / "training_summaries"
    summary_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    best_state = None
    patience = config["training"].get("early_stopping", 5)
    epochs_no_improve = 0

    start_time = time.time()
    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")

        train_loss = train_one_epoch(
            model, args.model_type, train_loader, optimizer, device, config,
            T, norm_stats, u_idx, v_idx, lambda_ke, ke_mode, ke_use_truth,
            ke_compare_to_last_input, do_assert=args.assert_shapes, use_physics=args.physics_loss
        )

        val_loss = validate(
            model, args.model_type, val_loader, device, config,
            T, norm_stats, u_idx, v_idx, lambda_ke, ke_mode, ke_use_truth,
            ke_compare_to_last_input, use_physics=args.physics_loss
        )

        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }
            print(f"New best model found at epoch {epoch}!")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs...")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {epoch} epochs (no improvement).")
            break

    # Save results
    total_time = time.time() - start_time
    if best_state:
        save_name = args.save_name or args.model_type
        ckpt_path = ckpt_dir / f"{save_name}.pt"
        torch.save(best_state, ckpt_path)
        print(f"\nTraining complete! Best model saved to {ckpt_path} (val_loss={best_val_loss:.4f})")
        print(f"Total training time: {total_time/60:.2f} min")

        summary_path = summary_dir / f"training_summary_{save_name}.txt"
        with open(summary_path, "w") as f:
            f.write(f"Training Summary for {args.model_type.upper()}\n")
            f.write(f"Best validation loss: {best_val_loss:.4f}\n")
            f.write(f"Total time: {total_time/60:.2f} min\n")
    else:
        print("\nNo model saved (training may have failed).")


if __name__ == "__main__":
    main()
