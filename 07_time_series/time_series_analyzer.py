"""
Time Series Analyzer
====================
Comprehensive toolkit for time series analysis, decomposition,
stationarity testing, forecasting models, and anomaly detection.

Author: Data Science Analytics Toolkit
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class StationarityTester:
    """Tests for stationarity using ADF, KPSS, and rolling statistics."""

    def adf_test(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller test for unit root.

        Parameters
        ----------
        series : pd.Series
            Time series to test.
        alpha : float
            Significance level (default 0.05).

        Returns
        -------
        dict
            Test statistic, p-value, critical values, and stationarity verdict.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ADF test.")
        result = adfuller(series.dropna(), autolag="AIC")
        output = {
            "test_statistic": result[0],
            "p_value": result[1],
            "n_lags": result[2],
            "n_observations": result[3],
            "critical_values": result[4],
            "is_stationary": result[1] < alpha,
            "interpretation": (
                "Series is stationary (reject H0)" if result[1] < alpha
                else "Series is non-stationary (fail to reject H0)"
            ),
        }
        return output

    def kpss_test(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """
        KPSS test for level or trend stationarity.

        Parameters
        ----------
        series : pd.Series
            Time series to test.
        alpha : float
            Significance level.

        Returns
        -------
        dict
            Test statistic, p-value, critical values, and stationarity verdict.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for KPSS test.")
        stat, p_value, n_lags, critical_values = kpss(series.dropna(), regression="c", nlags="auto")
        output = {
            "test_statistic": stat,
            "p_value": p_value,
            "n_lags": n_lags,
            "critical_values": critical_values,
            "is_stationary": p_value > alpha,
            "interpretation": (
                "Series is stationary (fail to reject H0)" if p_value > alpha
                else "Series is non-stationary (reject H0)"
            ),
        }
        return output

    def rolling_stats(
        self, series: pd.Series, window: int = 12
    ) -> pd.DataFrame:
        """
        Compute rolling mean and standard deviation to assess stationarity visually.

        Parameters
        ----------
        series : pd.Series
            Time series.
        window : int
            Rolling window size.

        Returns
        -------
        pd.DataFrame
            DataFrame with original series, rolling mean, and rolling std.
        """
        df = pd.DataFrame({"original": series})
        df["rolling_mean"] = series.rolling(window=window).mean()
        df["rolling_std"] = series.rolling(window=window).std()
        return df

    def full_report(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """Run both ADF and KPSS and summarize stationarity."""
        adf = self.adf_test(series, alpha)
        kpss_result = self.kpss_test(series, alpha)
        both_agree_stationary = adf["is_stationary"] and kpss_result["is_stationary"]
        return {
            "adf": adf,
            "kpss": kpss_result,
            "both_agree_stationary": both_agree_stationary,
            "summary": (
                "Both tests confirm stationarity." if both_agree_stationary
                else "Tests disagree or series is non-stationary — consider differencing."
            ),
        }


class TimeSeriesDecomposer:
    """Decompose time series into trend, seasonality, and residual components."""

    def classical_decompose(
        self,
        series: pd.Series,
        period: int,
        model: str = "additive",
    ) -> Any:
        """
        Classical seasonal decomposition (moving average).

        Parameters
        ----------
        series : pd.Series
            Time series with DatetimeIndex.
        period : int
            Period of the seasonality (e.g., 12 for monthly, 7 for weekly).
        model : str
            'additive' or 'multiplicative'.

        Returns
        -------
        DecomposeResult
            Statsmodels decomposition result with trend, seasonal, resid.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required.")
        return seasonal_decompose(series, period=period, model=model, extrapolate_trend="freq")

    def stl_decompose(
        self,
        series: pd.Series,
        period: int,
        robust: bool = True,
    ) -> Any:
        """
        STL (Seasonal-Trend decomposition using LOESS) — more robust than classical.

        Parameters
        ----------
        series : pd.Series
            Time series.
        period : int
            Seasonality period.
        robust : bool
            Use robust fitting to handle outliers.

        Returns
        -------
        STLForecast result object.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required.")
        stl = STL(series, period=period, robust=robust)
        return stl.fit()

    def extract_components(self, decomposition: Any) -> pd.DataFrame:
        """Extract trend, seasonal, and residual into a tidy DataFrame."""
        return pd.DataFrame(
            {
                "trend": decomposition.trend,
                "seasonal": decomposition.seasonal,
                "residual": decomposition.resid,
            }
        )


class ARIMAForecaster:
    """Fit and forecast using ARIMA and auto-order selection."""

    def __init__(self):
        self.model = None
        self.fitted = None
        self.order = None

    def fit(
        self,
        series: pd.Series,
        order: Tuple[int, int, int] = (1, 1, 1),
    ) -> "ARIMAForecaster":
        """
        Fit an ARIMA model.

        Parameters
        ----------
        series : pd.Series
            Univariate time series.
        order : tuple
            (p, d, q) order for ARIMA.

        Returns
        -------
        self
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required.")
        self.order = order
        self.model = ARIMA(series, order=order)
        self.fitted = self.model.fit()
        return self

    def forecast(self, steps: int = 12) -> pd.Series:
        """
        Generate future forecasts.

        Parameters
        ----------
        steps : int
            Number of periods to forecast.

        Returns
        -------
        pd.Series
            Forecasted values.
        """
        if self.fitted is None:
            raise ValueError("Model not fitted. Call fit() first.")
        forecast = self.fitted.forecast(steps=steps)
        return forecast

    def summary(self) -> str:
        """Return model summary as string."""
        if self.fitted is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return str(self.fitted.summary())

    def residual_diagnostics(self) -> Dict[str, float]:
        """Basic diagnostics on model residuals."""
        if self.fitted is None:
            raise ValueError("Model not fitted.")
        resid = self.fitted.resid
        return {
            "mean_residual": float(resid.mean()),
            "std_residual": float(resid.std()),
            "ljung_box_p": float(self.fitted.test_serial_correlation("ljungbox", lags=10)[0, 1]),
        }


class ExponentialSmoothingForecaster:
    """Holt-Winters Exponential Smoothing for trend + seasonality."""

    def __init__(self):
        self.fitted = None

    def fit(
        self,
        series: pd.Series,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: int = 12,
        damped_trend: bool = False,
    ) -> "ExponentialSmoothingForecaster":
        """
        Fit Holt-Winters model.

        Parameters
        ----------
        series : pd.Series
            Time series data.
        trend : str or None
            'add', 'mul', or None.
        seasonal : str or None
            'add', 'mul', or None.
        seasonal_periods : int
            Length of the seasonal cycle.
        damped_trend : bool
            Whether to damp the trend.

        Returns
        -------
        self
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required.")
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
        )
        self.fitted = model.fit(optimized=True)
        return self

    def forecast(self, steps: int = 12) -> pd.Series:
        """Forecast future values."""
        if self.fitted is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.fitted.forecast(steps)

    def smoothing_params(self) -> Dict[str, float]:
        """Return fitted smoothing parameters."""
        return {
            "alpha": self.fitted.params.get("smoothing_level"),
            "beta": self.fitted.params.get("smoothing_trend"),
            "gamma": self.fitted.params.get("smoothing_seasonal"),
            "phi": self.fitted.params.get("damping_trend"),
        }


class TimeSeriesAnomalyDetector:
    """Detect anomalies in time series using statistical methods."""

    def z_score_anomalies(
        self, series: pd.Series, threshold: float = 3.0
    ) -> pd.Series:
        """
        Flag anomalies where |z-score| > threshold.

        Parameters
        ----------
        series : pd.Series
            Time series.
        threshold : float
            Z-score cutoff.

        Returns
        -------
        pd.Series
            Boolean mask of anomaly positions.
        """
        z_scores = (series - series.mean()) / series.std()
        return z_scores.abs() > threshold

    def iqr_anomalies(self, series: pd.Series, factor: float = 1.5) -> pd.Series:
        """
        Flag anomalies outside Q1 - factor*IQR or Q3 + factor*IQR.

        Parameters
        ----------
        series : pd.Series
            Time series.
        factor : float
            IQR multiplier (1.5 = standard, 3.0 = extreme outliers only).

        Returns
        -------
        pd.Series
            Boolean mask of anomaly positions.
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return (series < lower) | (series > upper)

    def rolling_anomalies(
        self, series: pd.Series, window: int = 30, sigma: float = 2.5
    ) -> pd.Series:
        """
        Detect anomalies relative to a rolling mean ± sigma * rolling_std.

        Parameters
        ----------
        series : pd.Series
            Time series.
        window : int
            Rolling window size.
        sigma : float
            Number of standard deviations for the band.

        Returns
        -------
        pd.Series
            Boolean mask where True = anomaly.
        """
        rolling_mean = series.rolling(window=window, center=True).mean()
        rolling_std = series.rolling(window=window, center=True).std()
        upper_band = rolling_mean + sigma * rolling_std
        lower_band = rolling_mean - sigma * rolling_std
        return (series > upper_band) | (series < lower_band)

    def get_anomaly_summary(
        self, series: pd.Series, method: str = "zscore"
    ) -> pd.DataFrame:
        """
        Return a DataFrame of anomalous timestamps and values.

        Parameters
        ----------
        series : pd.Series
            Time series.
        method : str
            'zscore', 'iqr', or 'rolling'.

        Returns
        -------
        pd.DataFrame
        """
        if method == "zscore":
            mask = self.z_score_anomalies(series)
        elif method == "iqr":
            mask = self.iqr_anomalies(series)
        elif method == "rolling":
            mask = self.rolling_anomalies(series)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'zscore', 'iqr', or 'rolling'.")
        anomalies = series[mask]
        return pd.DataFrame({"timestamp": anomalies.index, "value": anomalies.values})


class TimeSeriesPipeline:
    """
    End-to-end time series pipeline:
      1. Stationarity testing
      2. Decomposition
      3. ARIMA forecasting
      4. Anomaly detection
    """

    def __init__(self, series: pd.Series, period: int = 12):
        self.series = series
        self.period = period
        self.tester = StationarityTester()
        self.decomposer = TimeSeriesDecomposer()
        self.forecaster = ARIMAForecaster()
        self.anomaly_detector = TimeSeriesAnomalyDetector()

    def run(self, forecast_steps: int = 12, arima_order: Tuple = (1, 1, 1)) -> Dict[str, Any]:
        """
        Execute the full pipeline.

        Returns
        -------
        dict
            Results from each stage.
        """
        results = {}

        # 1. Stationarity
        results["stationarity"] = self.tester.full_report(self.series)

        # 2. Decomposition
        decomp = self.decomposer.classical_decompose(self.series, period=self.period)
        results["decomposition"] = self.decomposer.extract_components(decomp)

        # 3. ARIMA Forecast
        self.forecaster.fit(self.series, order=arima_order)
        results["forecast"] = self.forecaster.forecast(steps=forecast_steps)

        # 4. Anomaly Detection
        results["anomalies"] = self.anomaly_detector.get_anomaly_summary(self.series)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate synthetic monthly time series
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=120, freq="MS")
    trend = np.linspace(100, 200, 120)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(120) / 12)
    noise = np.random.normal(0, 5, 120)
    series = pd.Series(trend + seasonality + noise, index=dates, name="sales")

    pipeline = TimeSeriesPipeline(series, period=12)
    results = pipeline.run(forecast_steps=12, arima_order=(1, 1, 1))

    print("=== Stationarity ===")
    print(results["stationarity"]["summary"])

    print("\n=== ARIMA Forecast (next 12 months) ===")
    print(results["forecast"].round(2).to_string())

    print("\n=== Anomalies Detected ===")
    print(results["anomalies"])
