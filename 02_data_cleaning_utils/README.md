# 🧹 Data Cleaning Utils — Utilidades de Limpieza de Datos

Librería Python reutilizable con funciones y clases para limpiar datos de forma estandarizada antes de entrenar modelos de Machine Learning.

## 📁 Estructura

```
02_data_cleaning_utils/
├── data_cleaning.py    # Librería principal de limpieza
└── README.md
```

## ⚙️ Funciones Incluidas

| Función | Descripción |
|---------|-------------|
| `handle_nulls(df, strategy)` | Imputa nulos con media, mediana o moda |
| `remove_duplicates(df)` | Elimina filas 100% duplicadas |
| `fix_dtypes(df)` | Convierte columnas a sus tipos correctos |
| `cap_outliers(df, cols)` | Aplica capping (IQR) a outliers extremos |
| `encode_categoricals(df)` | Codifica variables categóricas (Label / One-Hot) |
| `normalize_numerics(df)` | Escala variables numéricas (Standard / MinMax) |

## 🚀 Uso

```python
from data_cleaning import DataCleaner

cleaner = DataCleaner(df)
df_clean = cleaner.run_pipeline()
```
