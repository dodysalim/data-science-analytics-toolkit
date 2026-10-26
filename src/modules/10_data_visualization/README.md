# 10 — Kit de Visualización de Datos

Gráficos y dashboards de calidad profesional usando matplotlib y seaborn. Diseño SOLID con clases dedicadas por dominio de visualización.

## Autor

**Dody Dueñas**

## Clases

| Clase | Gráficos |
|---|---|
| `DistributionPlotter` | Histograma+KDE, Q-Q plot, violín/caja, distribuciones múltiples |
| `CorrelationPlotter` | Mapa de calor, pair plot, dispersión+regresión |
| `CategoricalPlotter` | Gráfico de barras, donut, barras apiladas |
| `TimeSeriesPlotter` | Línea + promedio móvil, box plots estacionales |
| `ModelPerformancePlotter` | Curva ROC, matriz de confusión, residuos, importancia de características |
| `DashboardBuilder` | Dashboard EDA multipanel automático |

## Uso

```python
from data_visualization import DistributionPlotter, DashboardBuilder

# Gráfico individual
dp = DistributionPlotter()
fig = dp.histogram_kde(df["ingresos"], title="Distribución de Ingresos")
fig.savefig("dist_ingresos.png", dpi=150)

# Dashboard EDA completo
builder = DashboardBuilder()
fig = builder.eda_dashboard(df, numeric_cols=["edad", "ingresos", "puntaje"], cat_col="segmento")
fig.savefig("dashboard_eda.png", dpi=150)
```

## Dependencias

```
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
pandas>=1.5.0
numpy>=1.23.0
```
