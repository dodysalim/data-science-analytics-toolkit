# 🧹 Data Cleaning Utilities

Librería Python con funciones reutilizables para **limpieza y preprocesamiento de datos**, diseñada para proyectos de Data Science y Machine Learning.

## ✨ Funcionalidades

| Módulo | Funciones |
|--------|-----------|
| **Nulos** | `report_nulls`, `fill_nulls_smart`, `drop_high_null_columns` |
| **Duplicados** | `remove_duplicates` |
| **Escalado** | `normalize_columns` (MinMax, StandardScaler, Log) |
| **Encoding** | `encode_categoricals` (One-Hot, Label) |
| **Texto** | `clean_text_column` |
| **Fechas** | `parse_dates` (extrae año, mes, día, etc.) |
| **Outliers** | `remove_outliers_iqr`, `cap_outliers` |
| **Pipeline** | `full_cleaning_pipeline` |

## 🚀 Uso Rápido

```python
from data_cleaning import full_cleaning_pipeline, report_nulls

# Ver nulos
print(report_nulls(df))

# Limpieza completa en un paso
df_clean = full_cleaning_pipeline(
    df,
    null_strategy='auto',     # media/moda según tipo
    encoding_method='label',  # codificación de categóricas
    normalize_method='minmax',# normalización [0,1]
    remove_dups=True          # eliminar duplicados
)
```

## 📦 Instalación

```bash
pip install pandas numpy scikit-learn
```

---
> **Autor:** Dody Dueñas | Data Analyst & Data Scientist
