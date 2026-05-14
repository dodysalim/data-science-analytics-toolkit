# 07 — Analizador de Series de Tiempo

Kit de herramientas completo para análisis de series de tiempo: pruebas de estacionariedad, descomposición, modelado ARIMA, suavizado exponencial y detección de anomalías.

## Autor

**Dody Dueñas**

## Módulos

| Clase | Descripción |
|---|---|
| `StationarityTester` | Pruebas ADF + KPSS, estadísticas de ventana móvil |
| `TimeSeriesDecomposer` | Descomposición clásica y STL (LOESS) |
| `ARIMAForecaster` | Ajuste y pronóstico con modelos ARIMA |
| `ExponentialSmoothingForecaster` | Suavizado exponencial triple de Holt-Winters |
| `TimeSeriesAnomalyDetector` | Detección de anomalías con Z-score, IQR y banda móvil |
| `TimeSeriesPipeline` | Pipeline de orquestación de extremo a extremo |

## Uso

```python
import pandas as pd
from time_series_analyzer import TimeSeriesPipeline

series = pd.read_csv("ventas.csv", index_col="fecha", parse_dates=True)["valor"]
pipeline = TimeSeriesPipeline(series, period=12)
resultados = pipeline.run(forecast_steps=24, arima_order=(2, 1, 2))

print(resultados["stationarity"]["summary"])
print(resultados["forecast"])
```

## Dependencias

```
statsmodels>=0.14.0
pandas>=1.5.0
numpy>=1.23.0
```
