# utils/losses.py
import torch
import torch.nn.functional as F


def _as_t(x, like):
    """Convert python float to torch tensor on same dtype as `like`."""
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


def _get_uv_stats(norm_stats, y_like, u_key=None, v_key=None):
    """
    norm_stats: dict like {"u10": (min,max), "v10": (min,max), ...}
    If u_key/v_key are not provided, we try to infer them by name.
    """
    if u_key is None or v_key is None:
        keys = list(norm_stats.keys())
        # guess that the keys starting with/including u and v are u and v
        def pick(cand, letter):
            # prefer exact startswith, then contains
            for k in cand:
                if k.lower().startswith(letter):
                    return k
            for k in cand:
                if letter in k.lower():
                    return k
            return None

        # only guess if not explicitly declared
        if u_key is None:
            u_key = pick(keys, "u")
        if v_key is None:
            v_key = pick(keys, "v")

    if u_key is None or v_key is None:
        raise ValueError(
            "Could not infer wind energy keys from normalization_stats. "
            "Pass explicit u_key/v_key to kinetic_energy_loss()."
        )

    u_min, u_max = norm_stats[u_key]
    v_min, v_max = norm_stats[v_key]

    # cast to tensors on correct device/dtype later
    return u_min, u_max, v_min, v_max


def _flatten_to_NC(t):
    """
    Accepts [N,C], [B,N,C], [C,H,W], or [B,C,H,W] and returns [N_total, C].
    """
    if t.dim() == 2:
        return t
    if t.dim() == 3:
        # [C,H,W] -> [H*W, C]
        C, H, W = t.shape
        return t.permute(1, 2, 0).reshape(H * W, C)
    if t.dim() == 4:
        # [B,C,H,W] or [B,N,C]
        B, C, H, W = t.shape
        return t.permute(0, 2, 3, 1).reshape(B * H * W, C)
    raise ValueError(f"Unsupported tensor shape {tuple(t.shape)}; expected [N,C], [B,N,C], [C,H,W], or [B,C,H,W].")


def kinetic_energy_loss(
    y_pred: torch.Tensor,
    y_ref: torch.Tensor,
    normalization_stats: dict,
    u_idx: int,
    v_idx: int,
    lambda_ke: float,
    mode: str = "pointwise",
    u_key: str | None = None,
    v_key: str | None = None,
) -> torch.Tensor:
    """
    Physics-based KE loss.

    Args:
        y_pred: predicted variables at forecast time. Commonly [N, C] (GNN path).
        y_ref:  reference variables to compare KE against (truth, or last input). Same shape as y_pred (at least in [*, C]).
        normalization_stats: dict {var_name: (min, max)} used for dataset min-max scaling.
        u_idx, v_idx: channel indices of u and v *within y_pred/y_ref*.
        lambda_ke: weight multiplier from config (e.g., 0.01).
        mode: "pointwise" | "global" | "both".
        u_key, v_key: optional explicit variable names to fetch stats from normalization_stats.
                      If omitted, we try to infer ("u10"/"v10" etc.).

    Returns:
        Scalar tensor = lambda_ke * KE_loss
    """
    # Flatten to [N_total, C]
    y_pred_nc = _flatten_to_NC(y_pred)
    y_ref_nc  = _flatten_to_NC(y_ref)

    if y_pred_nc.shape != y_ref_nc.shape:
        raise ValueError(f"y_pred and y_ref shapes must match after flatten. Got {tuple(y_pred_nc.shape)} vs {tuple(y_ref_nc.shape)}")

    # Fetch min/max for u and v from normalization stats
    u_min, u_max, v_min, v_max = _get_uv_stats(normalization_stats, y_pred_nc, u_key=u_key, v_key=v_key)

    # Make u and v tensors on the correct device/dtype
    u_min_t = _as_t(u_min, y_pred_nc)
    u_max_t = _as_t(u_max, y_pred_nc)
    v_min_t = _as_t(v_min, y_pred_nc)
    v_max_t = _as_t(v_max, y_pred_nc)

    # Extract normalized u,v
    u_pred_n = y_pred_nc[:, u_idx]
    v_pred_n = y_pred_nc[:, v_idx]
    u_ref_n  = y_ref_nc[:,  u_idx]
    v_ref_n  = y_ref_nc[:,  v_idx]

    # Denormalize (min-max)
    u_pred = u_min_t + u_pred_n * (u_max_t - u_min_t)
    v_pred = v_min_t + v_pred_n * (v_max_t - v_min_t)
    u_ref  = u_min_t + u_ref_n  * (u_max_t - u_min_t)
    v_ref  = v_min_t + v_ref_n  * (v_max_t - v_min_t)

    # Kinetic energy per node (per unit mass)
    ke_pred = 0.5 * (u_pred * u_pred + v_pred * v_pred)
    ke_ref  = 0.5 * (u_ref  * u_ref  + v_ref  * v_ref)

    # Compare
    if mode == "pointwise":
        ke_loss = F.mse_loss(ke_pred, ke_ref)
    elif mode == "global":
        ke_loss = F.mse_loss(ke_pred.mean(), ke_ref.mean())
    elif mode == "both":
        ke_loss = 0.5 * (F.mse_loss(ke_pred, ke_ref) + F.mse_loss(ke_pred.mean(), ke_ref.mean()))
    else:
        raise ValueError(f"Unknown KE mode: {mode}. Use 'pointwise', 'global', or 'both'.")

    return _as_t(lambda_ke, y_pred_nc) * ke_loss


# for reporting on physics loss
def kinetic_energy_metrics(
    y_pred: torch.Tensor,
    y_ref: torch.Tensor,
    normalization_stats: dict,
    u_idx: int,
    v_idx: int,
) -> dict[str, torch.Tensor]:
    """
    Returns two scalar tensors:
      - ke_pointwise_mse: MSE(KE_pred, KE_ref)
      - ke_global_mse:    MSE(mean(KE_pred), mean(KE_ref))
    """
    # reuse internal utilities
    y_pred_nc = _flatten_to_NC(y_pred)
    y_ref_nc  = _flatten_to_NC(y_ref)

    u_min, u_max, v_min, v_max = _get_uv_stats(normalization_stats, y_pred_nc)
    u_min_t = _as_t(u_min, y_pred_nc); u_max_t = _as_t(u_max, y_pred_nc)
    v_min_t = _as_t(v_min, y_pred_nc); v_max_t = _as_t(v_max, y_pred_nc)

    u_pred = u_min_t + y_pred_nc[:, u_idx] * (u_max_t - u_min_t)
    v_pred = v_min_t + y_pred_nc[:, v_idx] * (v_max_t - v_min_t)
    u_ref  = u_min_t + y_ref_nc[:,  u_idx] * (u_max_t - u_min_t)
    v_ref  = v_min_t + y_ref_nc[:,  v_idx] * (v_max_t - v_min_t)

    ke_pred = 0.5 * (u_pred*u_pred + v_pred*v_pred)
    ke_ref  = 0.5 * (u_ref*u_ref + v_ref*v_ref)

    ke_pointwise_mse = F.mse_loss(ke_pred, ke_ref)
    ke_global_mse    = F.mse_loss(ke_pred.mean(), ke_ref.mean())

    return {
        "ke_pointwise_mse": ke_pointwise_mse,
        "ke_global_mse": ke_global_mse,
    }

