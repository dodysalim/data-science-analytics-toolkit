# 10 — Data Visualization Toolkit

Publication-ready charts and dashboards using matplotlib and seaborn. SOLID-compliant design with dedicated classes per visualization domain.

## Classes

| Class | Charts |
|---|---|
| `DistributionPlotter` | Histogram+KDE, Q-Q plot, violin/box, multi-distribution |
| `CorrelationPlotter` | Heatmap, pair plot, scatter+regression |
| `CategoricalPlotter` | Bar chart, donut chart, stacked bar |
| `TimeSeriesPlotter` | Line + rolling average, seasonal box plots |
| `ModelPerformancePlotter` | ROC curve, confusion matrix, residuals, feature importance |
| `DashboardBuilder` | Auto EDA multi-panel dashboard |

## Usage

```python
from data_visualization import DistributionPlotter, DashboardBuilder

# Single plot
dp = DistributionPlotter()
fig = dp.histogram_kde(df["revenue"], title="Revenue Distribution")
fig.savefig("revenue_dist.png", dpi=150)

# Full EDA dashboard
builder = DashboardBuilder()
fig = builder.eda_dashboard(df, numeric_cols=["age", "income", "score"], cat_col="segment")
fig.savefig("eda_dashboard.png", dpi=150)
```

## Requirements

```
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
pandas>=1.5.0
numpy>=1.23.0
```
