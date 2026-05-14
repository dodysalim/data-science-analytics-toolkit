# 08 — Kit de Ingeniería de Características

Librería automatizada y manual de ingeniería de características siguiendo principios SOLID con API compatible con sklearn.

## Autor

**Dody Dueñas**

## Transformadores

| Clase | Propósito |
|---|---|
| `DateTimeFeatureExtractor` | Extrae año, mes, día, hora y codificaciones cíclicas sin/cos de columnas datetime |
| `PolynomialFeatureGenerator` | Genera características polinomiales e interacciones para columnas numéricas |
| `TargetEncoder` | Codificación de media objetivo suavizada para categóricas de alta cardinalidad |
| `FrequencyEncoder` | Codificación por frecuencia (proporción) para columnas categóricas |
| `Binner` | Discretización por ancho igual o cuantiles de características numéricas |
| `StatisticalFeatureSelector` | Selección de características por ANOVA, F-regresión o información mutua |
| `FeatureEngineeringPipeline` | Pipeline completo de orquestación de extremo a extremo |

## Uso

```python
from feature_engineering import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline(
    datetime_cols=["fecha_registro"],
    cat_cols=["categoria", "region"],
    num_cols=["edad", "ingresos"],
    poly_degree=2,
    n_bins=5,
    k_best=20,
    task="classification",
)

df_ingenieria = pipeline.fit_transform(df, y)
print(pipeline.get_feature_scores().head(10))
```

## Dependencias

```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
```
