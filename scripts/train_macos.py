import argparse
from pathlib import Path
import random  # added: for reproducible seeding
import numpy as np  # added: for reproducible seeding

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import yaml
from huggingface_hub import snapshot_download

from data.dataloader import get_dataloaders
from models.gtcnn import GTCNN
from utils.losses import kinetic_energy_loss, kinetic_energy_metrics  # added: physics based loss util


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models on ERA5 data")
    parser.add_argument(
        "--model_type",
        type=str,
        default="gtcnn",
        help="Model type to train",
    )
    parser.add_argument(
        "--assert_shapes",
        action="store_true",
        help="added: optional flag to verify tensor shapes on first batch",
    )  # added
    return parser.parse_args()


def pick_device():  # added: device selection
    if torch.backends.mps.is_available():  # added: for mac
        return torch.device("mps")
    elif torch.cuda.is_available():  # prev logic
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def seed_everything(seed: int = 42):  # added: deterministic seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# added: total loss computation including KE and standard MSE
def compute_total_loss(
    y_hat, batch, norm_stats, u_idx, v_idx,
    lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input,
    ref_override=None,  # NEW
):
    """
    Combines standard MSE and optional kinetic energy loss into one scalar.
    """
    loss_mse = F.mse_loss(y_hat, batch.y)
    loss_total = loss_mse

    if lambda_ke > 0:
        if ref_override is not None:
            ref = ref_override
        elif ke_use_truth:
            ref = batch.y
        else:
            ref = None  # if comparing to last input, must supply ref_override

        if ref is not None:
            loss_ke = kinetic_energy_loss(
                y_hat, ref, norm_stats, u_idx, v_idx, lambda_ke, mode=ke_mode
            )
            loss_total = loss_total + loss_ke

    return loss_total



def train_one_epoch(model, model_type, loader, optimizer, device, config, T, norm_stats,
                                     u_idx, v_idx, lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input, do_assert=False):  # changed: added T + do_assert args
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    first = True  # added: for single assert trigger

    for batch in progress_bar:
        optimizer.zero_grad()

        if model_type == "gtcnn":
            batch = batch.to(device)
            N = batch.y.size(0)  # same logic
            C_out = batch.y.size(1)

            # added: input array shape check
            if do_assert and first:
                assert batch.x.dim() == 2, f"Expected x [N*T, C_in], got {tuple(batch.x.shape)}"
                assert batch.y.dim() == 2, f"Expected y [N, C_out], got {tuple(batch.y.shape)}"
                first = False

            # added: AMP only if on CUDA
            use_amp = device.type == "cuda"
            if use_amp:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    y_hat = model(batch.x, batch.edge_index, N, T)
                    ref_override = None
                    if ke_compare_to_last_input:
                        start = (T - 1) * N
                        end = T * N
                        ref_override = batch.x[start:end, :C_out]  # last input slice, vars only
                    #loss = F.mse_loss(y_hat, batch.y)
                    loss_total = compute_total_loss(
                        y_hat, batch, norm_stats, u_idx, v_idx,
                        lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input,
                        ref_override=ref_override,
                    )

                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # added: plain FP32 path (for mac)
                y_hat = model(batch.x, batch.edge_index, N, T)
                ref_override = None
                if ke_compare_to_last_input:
                    start = (T - 1) * N
                    end = T * N
                    ref_override = batch.x[start:end, :C_out]  # last input slice, vars only
                #loss = F.mse_loss(y_hat, batch.y)
                loss_total = compute_total_loss(
                    y_hat, batch, norm_stats, u_idx, v_idx,
                    lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input,
                    ref_override=ref_override,
                )

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
def validate(model, model_type, loader, device, config, T, norm_stats,
                                     u_idx, v_idx, lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input):  # changed: added T arg
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Validating", leave=False)
    sum_ke_pointwise = 0.0
    sum_ke_global = 0.0
    for batch in progress_bar:
        if model_type == "gtcnn":
            batch = batch.to(device)
            N = batch.y.size(0)
            C_out = batch.y.size(1)

            y_hat = model(batch.x, batch.edge_index, N, T)

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

            ke_ref_for_metrics = batch.y  # we report vs truth regardless of training mode
            ke_metrics = kinetic_energy_metrics(y_hat, ke_ref_for_metrics, norm_stats, u_idx, v_idx)
            ke_pointwise = ke_metrics["ke_pointwise_mse"].item()
            ke_global = ke_metrics["ke_global_mse"].item()
            sum_ke_pointwise += ke_pointwise
            sum_ke_global += ke_global


        elif model_type == "cnn3d":
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss_total = F.mse_loss(y_hat, y)

        total_loss += loss_total.item()
        progress_bar.set_postfix({
            "loss": f"{loss_total.item():.4f}",
            "KEp": f"{ke_pointwise:.3e}",
            "KEg": f"{ke_global:.3e}",
        })
        avg_ke_pointwise = sum_ke_pointwise / max(1, len(loader))
        avg_ke_global = sum_ke_global / max(1, len(loader))
        print(f"Validation KE | pointwise: {avg_ke_pointwise:.3e} | global: {avg_ke_global:.3e}")

    return total_loss / len(loader)


def initialize_model(model_config, model_type, C_in, C_out):
    if model_type == "gtcnn":
        hidden_ch = model_config[model_type]["hidden_channels"]
        K = model_config[model_type]["chebyshev_order"]
        num_layers = model_config[model_type]["num_layers"]
        dropout = model_config[model_type]["dropout"]
        model = GTCNN(
            in_channels=C_in,
            hidden_channels=hidden_ch,
            out_channels=C_out,
            K=K,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "cnn3d":
        hidden_ch = model_config[model_type]["hidden_channels"]
        # model = CNN3D(in_channels=C_in, hidden_channels=hidden_ch, out_channels=C_out)
    else:
        raise ValueError(f"Unknown model type !!!")

    return model


def main():
    args = parse_args()
    seed_everything(42)  # added: det seeding

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
            allow_patterns="*",
        )
    else:
        print(f"Dataset already exists at {local_dir}, skipping download.")

    # Model category
    category = "graph_based" if args.model_type in model_config.get("graph_based", {}).keys() else "grid_based"
    model_config = model_config[category]

    # Dataloaders
    train_loader, val_loader = get_dataloaders(config=config, model_category=category, eval_mode=False)

    # added: handle Subset-wrapped datasets
    base_ds = getattr(train_loader.dataset, "dataset", train_loader.dataset)
    C_in = base_ds.in_channels
    C_out = base_ds.out_channels

    # added: get relevant information for KE loss, including normalization stats
    norm_stats = base_ds.normalization_stats
    level_to_idx = base_ds.level_to_idx
    u_idx = level_to_idx[config["data"]["wind_u_key"]]
    v_idx = level_to_idx[config["data"]["wind_v_key"]]

    # sanity checking u and v
    print("u_idx,v_idx =", u_idx, v_idx)
    print("u stats:", norm_stats[config["data"]["wind_u_key"]])
    print("v stats:", norm_stats[config["data"]["wind_v_key"]])

    lambda_ke = float(config["training"]["lambda_ke"])
    ke_mode = config["training"]["ke_mode"]
    ke_use_truth = config["training"]["ke_use_truth"]
    ke_compare_to_last_input = config["training"]["ke_compare_to_last_input"]

    T = int(config["graph"]["input_length"])  # added: stores T

    device = pick_device()  # replaced cuda-only line
    print("Using device:", device)

    # Model selection
    model = initialize_model(model_config=model_config, model_type=args.model_type, C_in=C_in, C_out=C_out)
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    # Prepare checkpoint directory
    ckpt_dir = root_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    best_state = None
    patience = config["training"].get("early_stopping", 5)
    epochs_no_improve = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")

        # changed: pass T and assert flag into train/validate
        train_loss = train_one_epoch(model, args.model_type, train_loader, optimizer, device, config, T, norm_stats,
                                     u_idx, v_idx, lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input,do_assert=args.assert_shapes )
        val_loss = validate(model, args.model_type, val_loader, device, config, T, norm_stats,
                                     u_idx, v_idx, lambda_ke, ke_mode, ke_use_truth, ke_compare_to_last_input)

        print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

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
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement in {patience}).")
            break

    if best_state is not None:
        ckpt_path = ckpt_dir / f"best_{args.model_type}.pt"
        torch.save(best_state, ckpt_path)
        print(f"\nTraining complete! Best model saved to {ckpt_path} (val_loss={best_val_loss:.4f})")
    else:
        print("\nNo model was saved (training may have failed).")


if __name__ == "__main__":
    main()
