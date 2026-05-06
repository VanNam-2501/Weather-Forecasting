"""
trainer.py  –  Training, validation, and early-stopping logic.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from config import Config


# ─────────────────────────── Early Stopping ─────────────────────────────────

class EarlyStopping:
    """
    Monitors a validation metric and saves the best model checkpoint.
    Stops training when the metric hasn't improved for `patience` epochs.
    """

    def __init__(
        self,
        checkpoint_path: str,
        patience: int = 10,
        delta: float = 1e-4,
        verbose: bool = True,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.patience        = patience
        self.delta           = delta
        self.verbose         = verbose

        self.counter    = 0
        self.best_score: Optional[float] = None
        self.best_val   = float("inf")
        self.early_stop = False

    def __call__(self, val_metric: float, model: nn.Module) -> None:
        score = -val_metric   # higher is better

        if self.best_score is None:
            self.best_score = score
            self._save(val_metric, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"    EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self._save(val_metric, model)
            self.counter = 0

    def _save(self, val_metric: float, model: nn.Module) -> None:
        if self.verbose:
            print(
                f"    ✓ Val metric improved "
                f"({self.best_val:.6f} → {val_metric:.6f}). Saving …"
            )
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_val = val_metric


# ─────────────────────────── Training loop ──────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x, future_target=y, teacher_forcing_ratio=teacher_forcing_ratio)
        loss  = criterion(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, avg_mae) on the given loader."""
    model.eval()
    total_loss = total_mae = 0.0

    for x, y in loader:
        x, y  = x.to(device), y.to(device)
        preds = model(x, future_target=None)
        total_loss += criterion(preds, y).item()
        total_mae  += mean_absolute_error(
            y.cpu().numpy().flatten(), preds.cpu().numpy().flatten()
        )

    return total_loss / len(loader), total_mae / len(loader)


# ─────────────────────────── Full training run ──────────────────────────────

def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
) -> dict:
    """
    Train a model with scheduled sampling + early stopping.

    Returns
    -------
    history : dict  –  lists of train_loss, val_loss, val_mae per epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    ckpt_path    = os.path.join(cfg.OUTPUT_DIR, f"best_{model_name}.pth")
    early_stop   = EarlyStopping(ckpt_path, cfg.ES_PATIENCE, cfg.ES_DELTA)

    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    print(f"\n{'═'*60}")
    print(f"  Training: {model_name}  |  device: {cfg.DEVICE}")
    print(f"{'═'*60}")

    for epoch in range(1, cfg.EPOCHS + 1):
        # Scheduled sampling: linearly decay teacher-forcing ratio to 0
        tf_ratio   = max(0.0, 1.0 - (epoch - 1) / cfg.TF_DECAY_EPOCHS)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.DEVICE, tf_ratio
        )
        val_loss, val_mae = evaluate(model, val_loader, criterion, cfg.DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        scheduler.step(val_loss)

        print(
            f"  Epoch {epoch:03d}/{cfg.EPOCHS}  "
            f"tf={tf_ratio:.2f}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_mae={val_mae:.4f}"
        )

        early_stop(val_mae, model)
        if early_stop.early_stop:
            print(f"  ⚑  Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
        print(f"  ✓ Best weights restored from {ckpt_path}")

    return history
