"""
train.py  –  Entry-point for training all weather-forecasting models.

Usage
-----
    python train.py                    # train all 3 models
    python train.py --model LSTM       # train a single model
    python train.py --eval-only        # skip training, run evaluation
    python train.py --model LSTM --debug   # fast smoke-test (2 epochs)
"""

import argparse
import os
import sys

import torch

from config import Config
from dataset import build_dataloaders
from evaluate import (
    compute_metrics,
    get_predictions,
    inverse_transform,
    plot_forecasts,
    plot_training_history,
    print_metrics_table,
)
from models import MODEL_REGISTRY, build_model, model_summary
from trainer import train_model


# ─────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train deep learning models for weather forecasting."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="Train a single model (default: train all).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run evaluation on saved checkpoints.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Quick smoke-test: 2 epochs, small batch size.",
    )
    return parser.parse_args()


# ─────────────────────────── Main ───────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = Config()

    # Debug overrides
    if args.debug:
        cfg.EPOCHS     = 2
        cfg.BATCH_SIZE = 16
        cfg.NUM_WORKERS = 0
        print("⚙  DEBUG mode: 2 epochs, batch=16")

    print(f"\n{'═'*60}")
    print(f"  Weather Forecasting with Deep Learning")
    print(f"  Device : {cfg.DEVICE}")
    print(f"  Models : {', '.join(MODEL_REGISTRY.keys())}")
    print(f"{'═'*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, scaler, target_idx, n_features = (
        build_dataloaders(cfg)
    )

    model_names = [args.model] if args.model else list(MODEL_REGISTRY.keys())

    histories:       dict = {}
    model_preds_inv: dict = {}
    model_metrics:   dict = {}
    y_true_inv             = None

    for model_name in model_names:

        # ── Build model ───────────────────────────────────────────────────────
        model = build_model(model_name, cfg, n_features, target_idx)
        model_summary(model, model_name)

        # ── Train ─────────────────────────────────────────────────────────────
        if not args.eval_only:
            history = train_model(
                model_name, model, train_loader, val_loader, cfg
            )
            histories[model_name] = history
        else:
            # Load existing checkpoint
            ckpt = os.path.join(cfg.OUTPUT_DIR, f"best_{model_name}.pth")
            if not os.path.exists(ckpt):
                print(f"  ⚠  No checkpoint found for {model_name} at {ckpt}. Skipping.")
                continue
            model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))
            print(f"  Loaded weights from {ckpt}")

        # ── Evaluate on test set ──────────────────────────────────────────────
        y_true_s, y_pred_s = get_predictions(model, test_loader, cfg.DEVICE)

        if y_true_inv is None:
            y_true_inv = inverse_transform(y_true_s, scaler, target_idx, n_features)

        y_pred_inv                  = inverse_transform(y_pred_s, scaler, target_idx, n_features)
        model_preds_inv[model_name] = y_pred_inv
        model_metrics[model_name]   = compute_metrics(y_true_inv, y_pred_inv)

    # ── Summary ───────────────────────────────────────────────────────────────
    if model_metrics:
        print_metrics_table(model_metrics)

    if y_true_inv is not None and model_preds_inv:
        plot_forecasts(y_true_inv, model_preds_inv, model_metrics)

    if histories:
        plot_training_history(histories)

    print("\n✓  Done.\n")


if __name__ == "__main__":
    main()
