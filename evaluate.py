"""
evaluate.py  –  Test-set evaluation and result visualisation.

------------------
    python evaluate.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from config import Config
from models import build_model, model_summary


# ─────────────────────────── Inference helpers ──────────────────────────────

@torch.no_grad()
def get_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over the full loader (no teacher forcing).

    Returns
    -------
    y_true, y_pred : np.ndarray  shape (N, H)
    """
    model.eval()
    trues, preds = [], []

    for x, y in loader:
        x    = x.to(device)
        pred = model(x, future_target=None).cpu().numpy()
        preds.append(pred)
        trues.append(y.numpy())

    return (
        np.concatenate(trues, axis=0).squeeze(-1),   # (N, H)
        np.concatenate(preds, axis=0).squeeze(-1),   # (N, H)
    )


def inverse_transform(
    scaled: np.ndarray,
    scaler,
    target_idx: int,
    n_features: int,
) -> np.ndarray:
    """
    Invert StandardScaler for the target column only.

    Parameters
    ----------
    scaled : np.ndarray  shape (N, H)
    """
    N, H = scaled.shape
    flat  = scaled.flatten()
    dummy = np.zeros((len(flat), n_features))
    dummy[:, target_idx] = flat
    inv   = scaler.inverse_transform(dummy)[:, target_idx]
    return inv.reshape(N, H)


# ─────────────────────────── Metrics table ──────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE / RMSE / MSE over the full test set (all horizons)."""
    flat_true = y_true.flatten()
    flat_pred = y_pred.flatten()
    mae  = mean_absolute_error(flat_true, flat_pred)
    mse  = mean_squared_error(flat_true, flat_pred)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "RMSE": rmse, "MSE": mse}


def print_metrics_table(metrics_dict: dict[str, dict]) -> None:
    """Pretty-print a comparison table of all models."""
    rows = [
        {"Model": name, **m}
        for name, m in metrics_dict.items()
    ]
    df = pd.DataFrame(rows).set_index("Model")
    pd.options.display.float_format = "{:,.4f}".format
    print("\n" + "─" * 50)
    print("  TEST-SET METRICS")
    print("─" * 50)
    print(df.to_string())
    print("─" * 50 + "\n")


# ─────────────────────────── Plotting ───────────────────────────────────────

COLORS = {"LSTM": "#1f77b4", "LSTM_Attn": "#ff7f0e", "Transformer": "#2ca02c"}
STEPS_TO_PLOT = [0, 2, 5, 8, 10, 11]   # forecast horizons to visualise
ZOOM = slice(0, 100)                    # number of test samples to show


def plot_forecasts(
    y_true_inv: np.ndarray,
    model_preds_inv: dict[str, np.ndarray],
    model_metrics: dict[str, dict],
    save_path: str = "logs/forecast_comparison.png",
) -> None:
    """
    Plot actual vs. predicted for several forecast horizons.
    Saves the figure to `save_path`.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, step in enumerate(STEPS_TO_PLOT):
        ax = axes[i]
        ax.plot(
            y_true_inv[ZOOM, step],
            "k-o", markersize=3, alpha=0.4,
            label="Actual" if i == 0 else None,
        )

        for name, preds in model_preds_inv.items():
            mae = model_metrics[name]["MAE"]
            lbl = f"{name} (MAE={mae:.4f})" if i == 0 else f"MAE={mae:.2f}"
            ax.plot(
                preds[ZOOM, step],
                "--", color=COLORS.get(name, "red"), linewidth=1.5, label=lbl,
            )

        ax.set_title(f"Forecast +{step + 1}h", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Forecast plot saved → {save_path}")
    plt.show()


def plot_training_history(
    histories: dict[str, dict],
    save_path: str = "logs/training_history.png",
) -> None:
    """Plot train/val loss and val MAE curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories.items():
        color = COLORS.get(name, "gray")
        line, = axes[0].plot(hist["train_loss"], color=color, label=f"{name} train")
        axes[0].plot(hist["val_loss"], "--", color=color, label=f"{name} val")
        axes[1].plot(hist["val_mae"], color=color, marker=".", label=name)

    axes[0].set_title("MSE Loss: Train (—) vs Val (- -)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation MAE (↓ better)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Training history plot saved → {save_path}")
    plt.show()


# ─────────────────────────── Standalone evaluation ──────────────────────────

if __name__ == "__main__":
    from dataset import build_dataloaders
    from models import MODEL_REGISTRY

    cfg = Config()
    _, _, test_loader, scaler, target_idx, n_features = build_dataloaders(cfg)

    model_preds_inv: dict[str, np.ndarray] = {}
    model_metrics:   dict[str, dict]       = {}
    y_true_inv: np.ndarray | None          = None

    for model_name in MODEL_REGISTRY:
        ckpt = os.path.join(cfg.OUTPUT_DIR, f"best_{model_name}.pth")
        if not os.path.exists(ckpt):
            print(f"  ⚠  Checkpoint not found for {model_name}: {ckpt}")
            continue

        model = build_model(model_name, cfg, n_features, target_idx)
        model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))
        model_summary(model, model_name)

        y_true_s, y_pred_s = get_predictions(model, test_loader, cfg.DEVICE)

        if y_true_inv is None:
            y_true_inv = inverse_transform(y_true_s, scaler, target_idx, n_features)

        y_pred_inv                  = inverse_transform(y_pred_s, scaler, target_idx, n_features)
        model_preds_inv[model_name] = y_pred_inv
        model_metrics[model_name]   = compute_metrics(y_true_inv, y_pred_inv)

    if model_metrics:
        print_metrics_table(model_metrics)
        if y_true_inv is not None:
            plot_forecasts(y_true_inv, model_preds_inv, model_metrics)
