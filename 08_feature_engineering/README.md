# 08 — Feature Engineering Toolkit

Automated and manual feature engineering library following SOLID principles and sklearn-compatible API.

## Transformers

| Class | Purpose |
|---|---|
| `DateTimeFeatureExtractor` | Extract year, month, day, hour, cyclical sin/cos encodings from datetime columns |
| `PolynomialFeatureGenerator` | Generate polynomial and interaction features for numerical columns |
| `TargetEncoder` | Smoothed mean target encoding for high-cardinality categoricals |
| `FrequencyEncoder` | Frequency (proportion) encoding for categorical columns |
| `Binner` | Equal-width or quantile-based discretization of numeric features |
| `StatisticalFeatureSelector` | ANOVA, F-regression, or mutual information based feature selection |
| `FeatureEngineeringPipeline` | Full end-to-end orchestration pipeline |

## Usage

```python
from feature_engineering import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline(
    datetime_cols=["signup_date"],
    cat_cols=["category", "region"],
    num_cols=["age", "income"],
    poly_degree=2,
    n_bins=5,
    k_best=20,
    task="classification",
)

df_engineered = pipeline.fit_transform(df, y)
print(pipeline.get_feature_scores().head(10))
```

## Requirements

```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
```
