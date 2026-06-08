"""Stock-market dataset utilities for Part 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch.utils.data import DataLoader, Dataset

from parameters import FeatureMode, StockDataParams, TrainParams


def download_stock_data(
    data_params: StockDataParams,
    cache_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Download or load cached OHLC stock data from Yahoo Finance.

    Args:
        data_params: Tickers and date range for the download.
        cache_dir: Directory used to store CSV cache files.

    Returns:
        A mapping from ticker symbol to a chronologically sorted DataFrame.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, pd.DataFrame] = {}

    for ticker in data_params.tickers:
        cache_path = cache_dir / f"{ticker}.csv"
        if cache_path.exists():
            frame = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        else:
            raw = yf.download(
                ticker,
                start=data_params.start_date,
                end=data_params.test_end,
                auto_adjust=False,
                progress=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            frame = raw[["Open", "High", "Low", "Close"]].copy()
            frame.index.name = "Date"
            frame.to_csv(cache_path)

        data[ticker] = frame.sort_index()

    return data


def _split_mask(
    index: pd.DatetimeIndex,
    data_params: StockDataParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create boolean masks for chronological train/validation/test splits."""
    train_end = pd.Timestamp(data_params.train_end)
    val_end = pd.Timestamp(data_params.val_end)
    return (
        index <= train_end,
        (index > train_end) & (index <= val_end),
        index > val_end,
    )


def _add_auxiliary_features(features: np.ndarray) -> np.ndarray:
    """Append short moving averages of the close price."""
    close = features[:, 3]
    ma5 = pd.Series(close).rolling(5, min_periods=1).mean().to_numpy()
    ma10 = pd.Series(close).rolling(10, min_periods=1).mean().to_numpy()
    return np.column_stack([features, ma5, ma10])


def _normalize_features(
    features: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Normalize features using training-set mean and standard deviation."""
    train_values = features[train_mask]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std < 1e-8] = 1.0
    return (features - mean) / std


def _future_return(
    close: np.ndarray,
    high: np.ndarray,
    time_idx: int,
    horizon: int,
    mode: FeatureMode,
) -> float:
    """Compute a future return ratio for one horizon."""
    base = close[time_idx]
    future_price = high[time_idx + horizon] if mode == "turning_point" else close[time_idx + horizon]
    return (future_price - base) / base


def _rolling_average_target(
    close: np.ndarray,
    time_idx: int,
    horizon: int,
    window: int,
) -> float:
    """Compute the weighted rolling-average return target."""
    base = close[time_idx]
    future_prices = close[time_idx + horizon - window : time_idx + horizon + 1]
    weights = np.arange(1, len(future_prices) + 1, dtype=np.float64)
    weights /= weights.sum()
    weighted_price = np.dot(weights, future_prices)
    return (weighted_price - base) / base


def build_stock_sequences(
    stock_data: dict[str, pd.DataFrame],
    data_params: StockDataParams,
    mode: FeatureMode = "returns",
) -> dict[str, dict[str, np.ndarray]]:
    """Create sliding-window datasets for each chronological split.

    Args:
        stock_data: Raw stock price data keyed by ticker.
        data_params: Feature-engineering and split configuration.
        mode: Target type for the experiment.

    Returns:
        Dictionaries keyed by ``train``, ``val``, and ``test`` containing ``X`` and ``y`` arrays.
    """
    datasets: dict[str, dict[str, np.ndarray]] = {
        "train": {"X": [], "y": []},
        "val": {"X": [], "y": []},
        "test": {"X": [], "y": []},
    }

    for _, frame in stock_data.items():
        values = frame[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float64)
        close = values[:, 3]
        high = values[:, 1]
        index = frame.index

        features = _add_auxiliary_features(values)
        train_mask, _, _ = _split_mask(index, data_params)
        features = _normalize_features(features, train_mask)

        max_t = len(values) - data_params.horizons - 1
        for time_idx in range(data_params.lookback - 1, max_t + 1):
            window = features[time_idx - data_params.lookback + 1 : time_idx + 1]

            if mode == "turning_point":
                future_returns = [
                    _future_return(close, high, time_idx, horizon, mode)
                    for horizon in range(1, data_params.horizons + 1)
                ]
                target = np.array(
                    [float(any(value > data_params.turning_point_gamma for value in future_returns))],
                    dtype=np.float32,
                )
            elif mode == "rolling_avg":
                target = np.array(
                    [
                        _rolling_average_target(close, time_idx, horizon, data_params.rolling_window)
                        for horizon in range(1, data_params.horizons + 1)
                    ],
                    dtype=np.float32,
                )
            else:
                target = np.array(
                    [
                        _future_return(close, high, time_idx, horizon, mode)
                        for horizon in range(1, data_params.horizons + 1)
                    ],
                    dtype=np.float32,
                )

            sample_day = index[time_idx]
            if sample_day <= pd.Timestamp(data_params.train_end):
                split_name = "train"
            elif sample_day <= pd.Timestamp(data_params.val_end):
                split_name = "val"
            else:
                split_name = "test"

            datasets[split_name]["X"].append(window.astype(np.float32))
            datasets[split_name]["y"].append(target)

    for split_name in datasets:
        datasets[split_name]["X"] = np.asarray(datasets[split_name]["X"], dtype=np.float32)
        datasets[split_name]["y"] = np.asarray(datasets[split_name]["y"], dtype=np.float32)

    return datasets


class StockSequenceDataset(Dataset):
    """PyTorch dataset wrapping Part 1 sequence tensors."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Store sequence features and targets.

        Args:
            features: Input array of shape ``(N, T, F)``.
            targets: Target array of shape ``(N, D)`` or ``(N, 1)``.
        """
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one feature/target pair."""
        return self.features[idx], self.targets[idx]


def make_stock_loaders(
    datasets: dict[str, dict[str, np.ndarray]],
    train_params: TrainParams,
) -> dict[str, DataLoader]:
    """Create DataLoader objects for Part 1 sequence data.

    Args:
        datasets: Output of :func:`build_stock_sequences`.
        train_params: Batch-size configuration.

    Returns:
        Data loaders for ``train``, ``val``, and ``test``.
    """
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = StockSequenceDataset(datasets[split]["X"], datasets[split]["y"])
        loaders[split] = DataLoader(
            dataset,
            batch_size=train_params.batch_size,
            shuffle=split == "train",
            drop_last=False,
        )
    return loaders
