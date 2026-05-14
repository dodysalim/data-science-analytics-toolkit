# 07 — Time Series Analyzer

Comprehensive toolkit for time series analysis, decomposition, stationarity testing, ARIMA forecasting, and anomaly detection.

## Modules

| Class | Description |
|---|---|
| `StationarityTester` | ADF + KPSS tests, rolling statistics |
| `TimeSeriesDecomposer` | Classical and STL decomposition |
| `ARIMAForecaster` | ARIMA model fitting and forecasting |
| `ExponentialSmoothingForecaster` | Holt-Winters triple exponential smoothing |
| `TimeSeriesAnomalyDetector` | Z-score, IQR, rolling band anomaly detection |
| `TimeSeriesPipeline` | End-to-end orchestration pipeline |

## Usage

```python
import pandas as pd
from time_series_analyzer import TimeSeriesPipeline

series = pd.read_csv("sales.csv", index_col="date", parse_dates=True)["value"]
pipeline = TimeSeriesPipeline(series, period=12)
results = pipeline.run(forecast_steps=24, arima_order=(2, 1, 2))

print(results["stationarity"]["summary"])
print(results["forecast"])
```

## Requirements

```
statsmodels>=0.14.0
pandas>=1.5.0
numpy>=1.23.0
```
