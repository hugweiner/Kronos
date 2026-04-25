"""Utility functions for the Kronos time series prediction library."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from datetime import datetime, timedelta
import warnings


def normalize_time_series(
    data: Union[np.ndarray, pd.Series, List[float]],
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, dict]:
    """Normalize a time series array.

    Args:
        data: Input time series data.
        method: Normalization method — 'minmax' or 'zscore'.
        feature_range: Target range for minmax normalization.

    Returns:
        Tuple of (normalized_array, params_dict) where params_dict contains
        the values needed to invert the transformation.
    """
    arr = np.array(data, dtype=np.float64).flatten()

    if method == "minmax":
        data_min = arr.min()
        data_max = arr.max()
        scale = data_max - data_min
        if scale == 0:
            warnings.warn("Constant series detected; returning zeros.")
            return np.zeros_like(arr), {"method": method, "min": data_min, "max": data_max}
        lo, hi = feature_range
        normalized = (arr - data_min) / scale * (hi - lo) + lo
        params = {"method": method, "min": data_min, "max": data_max, "range": feature_range}

    elif method == "zscore":
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            warnings.warn("Zero standard deviation; returning zeros.")
            return np.zeros_like(arr), {"method": method, "mean": mean, "std": std}
        normalized = (arr - mean) / std
        params = {"method": method, "mean": mean, "std": std}

    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Use 'minmax' or 'zscore'.")

    return normalized, params


def denormalize_time_series(data: np.ndarray, params: dict) -> np.ndarray:
    """Invert a normalization transform using the stored parameters.

    Args:
        data: Normalized array to invert.
        params: Parameter dict returned by :func:`normalize_time_series`.

    Returns:
        Array in the original scale.
    """
    arr = np.array(data, dtype=np.float64)
    method = params["method"]

    if method == "minmax":
        lo, hi = params["range"]
        scale = params["max"] - params["min"]
        return (arr - lo) / (hi - lo) * scale + params["min"]

    elif method == "zscore":
        return arr * params["std"] + params["mean"]

    raise ValueError(f"Unknown method in params: '{method}'.")


def create_sliding_windows(
    series: np.ndarray,
    window_size: int,
    horizon: int = 1,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create (X, y) sliding-window pairs from a 1-D time series.

    Args:
        series: 1-D array of values.
        window_size: Number of past observations per sample.
        horizon: Number of future steps to predict.
        step: Stride between consecutive windows.

    Returns:
        Tuple (X, y) with shapes (n_samples, window_size) and
        (n_samples, horizon).
    """
    series = np.array(series, dtype=np.float64).flatten()
    X, y = [], []
    total = len(series)
    for start in range(0, total - window_size - horizon + 1, step):
        X.append(series[start : start + window_size])
        y.append(series[start + window_size : start + window_size + horizon])
    return np.array(X), np.array(y)


def align_dates(
    df: pd.DataFrame,
    date_col: str = "date",
    freq: str = "B",
    fill_method: Optional[str] = "ffill",
) -> pd.DataFrame:
    """Reindex a DataFrame to a regular date frequency, filling gaps.

    Args:
        df: Input DataFrame with a date column.
        date_col: Name of the column containing dates.
        freq: Pandas frequency string (e.g. 'B' for business days).
        fill_method: How to fill missing rows — 'ffill', 'bfill', or None.

    Returns:
        DataFrame reindexed to the specified frequency.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_range)
    df.index.name = date_col
    if fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()
    return df.reset_index()


def compute_returns(prices: pd.Series, log: bool = False) -> pd.Series:
    """Compute simple or log returns from a price series.

    Args:
        prices: Series of asset prices.
        log: If True, compute log returns; otherwise simple returns.

    Returns:
        Series of returns (first value is NaN).
    """
    if log:
        return np.log(prices / prices.shift(1))
    return prices.pct_change()
