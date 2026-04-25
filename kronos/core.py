"""Core module for Kronos time-series prediction framework.

Provides the main KronosPredictor class that wraps the underlying
time-series model with a clean interface for stock market forecasting.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List
from datetime import datetime, timedelta


class KronosPredictor:
    """Main predictor class for time-series forecasting using the Kronos model.

    Attributes:
        model: The underlying forecasting model.
        context_length (int): Number of historical data points used for prediction.
        prediction_length (int): Number of future steps to predict.
        freq (str): Data frequency string (e.g., 'D' for daily, 'B' for business days).
    """

    def __init__(
        self,
        context_length: int = 512,
        prediction_length: int = 64,
        freq: str = "B",
        device: str = "cpu",
        model_size: str = "small",
    ):
        """
        Initialize the KronosPredictor.

        Args:
            context_length: Number of historical time steps to use as context.
            prediction_length: Number of future time steps to forecast.
            freq: Pandas-compatible frequency string for the time series.
            device: Compute device ('cpu' or 'cuda').
            model_size: Model variant to load ('tiny', 'small', 'base', 'large').
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq
        self.device = device
        self.model_size = model_size
        self.model = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Load the Kronos model from HuggingFace hub."""
        try:
            from kronos_forecaster import KronosPipeline  # type: ignore

            model_name = f"amazon/chronos-t5-{self.model_size}"
            self.model = KronosPipeline.from_pretrained(
                model_name,
                device_map=self.device,
            )
            self._is_loaded = True
        except ImportError:
            raise ImportError(
                "kronos_forecaster package is required. "
                "Install it with: pip install kronos-forecaster"
            )

    def prepare_context(
        self, series: pd.Series, context_length: Optional[int] = None
    ) -> np.ndarray:
        """Prepare context window from a pandas Series.

        Args:
            series: Time-indexed pandas Series of price/value data.
            context_length: Override instance context_length if provided.

        Returns:
            Numpy array of the most recent `context_length` values.
        """
        ctx_len = context_length or self.context_length
        values = series.dropna().values
        if len(values) < ctx_len:
            # Pad with NaN on the left if not enough history
            pad = np.full(ctx_len - len(values), np.nan)
            values = np.concatenate([pad, values])
        return values[-ctx_len:].astype(np.float32)

    def predict(
        self,
        series: pd.Series,
        num_samples: int = 20,
        quantile_levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Generate probabilistic forecasts for a time series.

        Args:
            series: Historical time series as a pandas Series with DatetimeIndex.
            num_samples: Number of sample paths to draw from the model.
            quantile_levels: Quantiles to compute (default: [0.1, 0.5, 0.9]).

        Returns:
            DataFrame with forecast quantiles indexed by future dates.
        """
        if not self._is_loaded:
            self.load_model()

        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]

        context = self.prepare_context(series)
        context_tensor = context[np.newaxis, :]  # shape: (1, context_length)

        forecast = self.model.predict(
            context=context_tensor,
            prediction_length=self.prediction_length,
            num_samples=num_samples,
        )

        # Build future date index
        last_date = series.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(self.freq),
            periods=self.prediction_length,
            freq=self.freq,
        )

        # Compute quantiles from samples (forecast shape: (1, num_samples, pred_len))
        samples = forecast[0]  # shape: (num_samples, prediction_length)
        quantile_df = pd.DataFrame(index=future_dates)
        for q in quantile_levels:
            col_name = f"q{int(q * 100):02d}"
            quantile_df[col_name] = np.quantile(samples, q, axis=0)

        quantile_df["mean"] = samples.mean(axis=0)
        return quantile_df

    def __repr__(self) -> str:
        return (
            f"KronosPredictor(model_size={self.model_size!r}, "
            f"context_length={self.context_length}, "
            f"prediction_length={self.prediction_length}, "
            f"freq={self.freq!r}, loaded={self._is_loaded})"
        )
