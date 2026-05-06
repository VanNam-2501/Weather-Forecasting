"""
dataset.py  –  Data pipeline for Jena Climate Weather Forecasting.

Responsibilities
----------------
* Download & cache the raw CSV (if needed).
* Feature engineering (wind vector decomposition, cyclic time encoding).
* Train / Val / Test split + Standard-scaling.
* PyTorch Dataset + DataLoader helpers.
"""

import os
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import Config


# ─────────────────────────── Dataset ────────────────────────────────────────

class WeatherDataset(Dataset):
    """Sliding-window dataset for multi-step time-series forecasting.

    Parameters
    ----------
    data : np.ndarray  shape (T, F)  – already scaled
    input_window : int
    forecast_horizon : int
    target_col_idx : int  – column index of the target variable
    """

    def __init__(
        self,
        data: np.ndarray,
        input_window: int,
        forecast_horizon: int,
        target_col_idx: int,
    ) -> None:
        self.data = torch.FloatTensor(data)
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.target_col_idx = target_col_idx

    def __len__(self) -> int:
        return len(self.data) - self.input_window - self.forecast_horizon + 1

    def __getitem__(self, idx: int):
        past = self.data[idx : idx + self.input_window]                        # (W, F)
        future = self.data[
            idx + self.input_window : idx + self.input_window + self.forecast_horizon,
            self.target_col_idx,
        ]                                                                       # (H,)
        return past, future.unsqueeze(-1)                                       # (W, F), (H, 1)


# ─────────────────────────── Helpers ────────────────────────────────────────

def _download_data(cfg: Config) -> None:
    """Download & extract the Jena Climate CSV if not present."""
    if os.path.exists(cfg.CSV_PATH):
        return

    os.makedirs(os.path.dirname(cfg.CSV_PATH), exist_ok=True)
    zip_path = cfg.CSV_PATH.replace(".csv", ".zip")

    print(f"Downloading dataset from {cfg.DATA_URL} …")
    os.system(f'curl -L -o "{zip_path}" "{cfg.DATA_URL}"')

    print("Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.dirname(cfg.CSV_PATH))

    # The zip extracts to jena_climate_2009_2016.csv – rename/move if needed
    extracted = os.path.join(os.path.dirname(cfg.CSV_PATH), "jena_climate_2009_2016.csv")
    if extracted != cfg.CSV_PATH and os.path.exists(extracted):
        os.rename(extracted, cfg.CSV_PATH)

    os.remove(zip_path)


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Sub-sample to hourly, add wind-vector & cyclic time features."""
    # Take every 6th row → hourly
    df = df[5::6].reset_index(drop=True)

    # Wind vector decomposition
    wv    = df.pop("wv (m/s)")
    wd_rad = df.pop("wd (deg)") * np.pi / 180
    df["Wx"] = wv * np.cos(wd_rad)
    df["Wy"] = wv * np.sin(wd_rad)

    # Cyclic time encoding
    timestamp = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S").map(
        pd.Timestamp.timestamp
    )
    day  = 24 * 60 * 60
    year = 365.2425 * 24 * 3600
    df["Day sin"]  = np.sin(timestamp * (2 * np.pi / day))
    df["Day cos"]  = np.cos(timestamp * (2 * np.pi / day))
    df["Year sin"] = np.sin(timestamp * (2 * np.pi / year))
    df["Year cos"] = np.cos(timestamp * (2 * np.pi / year))

    return df


# ─────────────────────────── Public API ─────────────────────────────────────

def build_dataloaders(cfg: Config):
    """
    Build and return train / val / test DataLoaders together with the
    fitted StandardScaler and target column index.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    scaler : StandardScaler
    target_idx : int
    n_features : int
    """
    _download_data(cfg)

    df = pd.read_csv(cfg.CSV_PATH)
    df = _feature_engineering(df)

    df_final    = df[cfg.FEATURE_COLS].copy()
    target_idx  = cfg.FEATURE_COLS.index(cfg.TARGET_COL)
    n_features  = len(cfg.FEATURE_COLS)

    # Train 80 % | Val 10 % | Test 10 %
    n = len(df_final)
    cut1, cut2 = int(n * 0.8), int(n * 0.9)

    scaler = StandardScaler()
    train_arr = scaler.fit_transform(df_final.iloc[:cut1].values)
    val_arr   = scaler.transform(df_final.iloc[cut1:cut2].values)
    test_arr  = scaler.transform(df_final.iloc[cut2:].values)

    def _make_loader(arr: np.ndarray, shuffle: bool = False) -> DataLoader:
        ds = WeatherDataset(arr, cfg.INPUT_WINDOW, cfg.FORECAST_HORIZON, target_idx)
        return DataLoader(
            ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=(cfg.DEVICE.type == "cuda"),
        )

    train_loader = _make_loader(train_arr, shuffle=True)
    val_loader   = _make_loader(val_arr)
    test_loader  = _make_loader(test_arr)

    print(
        f"Dataset ready  →  train: {len(train_arr):,} | "
        f"val: {len(val_arr):,} | test: {len(test_arr):,} samples"
    )
    print(f"Features ({n_features}): {cfg.FEATURE_COLS}")

    return train_loader, val_loader, test_loader, scaler, target_idx, n_features
