"""
Configuration file for Weather Forecasting with Deep Learning.
Centralizes all hyperparameters and paths.
"""

import torch


class Config:
    # ─── Data ────────────────────────────────────────────────────────────────
    CSV_PATH       = "data/jena_climate_2009_2016.csv"
    DATA_URL       = (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        "jena_climate_2009_2016.csv.zip"
    )
    TARGET_COL     = "T (degC)"
    FEATURE_COLS   = [
        "p (mbar)", "T (degC)", "rho (g/m**3)",
        "Wx", "Wy",
        "Day sin", "Day cos",
        "Year sin", "Year cos",
    ]

    # ─── Window / Horizon ────────────────────────────────────────────────────
    INPUT_WINDOW      = 96    # look-back steps (hours)
    FORECAST_HORIZON  = 12   # prediction horizon (hours)

    # ─── Model ───────────────────────────────────────────────────────────────
    D_MODEL    = 32
    NUM_LAYERS = 2
    DROPOUT    = 0.3
    NHEAD      = 4            # Transformer only

    # ─── Training ────────────────────────────────────────────────────────────
    BATCH_SIZE     = 64
    EPOCHS         = 120
    LEARNING_RATE  = 1e-4
    NUM_WORKERS    = 2

    # Early Stopping
    ES_PATIENCE = 10
    ES_DELTA    = 1e-4

    # Scheduled Sampling decay (teacher-forcing ratio → 0 over N epochs)
    TF_DECAY_EPOCHS = 20

    # ─── Paths ───────────────────────────────────────────────────────────────
    OUTPUT_DIR     = "checkpoints"
    LOG_DIR        = "logs"

    # ─── Device ──────────────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
