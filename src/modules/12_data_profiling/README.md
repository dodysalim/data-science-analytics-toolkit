# 12 — Generador de Perfiles de Datos y Reporte de Calidad

Perfilado automático de conjuntos de datos con estadísticas exhaustivas por columna, análisis de valores faltantes, detección de duplicados y puntuación de calidad de datos de 0 a 100 con calificación de letra.

## Autor

**Dody Dueñas**

## Clases

| Clase | Descripción |
|---|---|
| `NumericProfiler` | Media, mediana, desv. estándar, IQR, asimetría, curtosis, conteo de valores atípicos, forma de distribución |
| `CategoricalProfiler` | Cardinalidad, valores principales, entropía, detección de binarios e identificadores |
| `DatetimeProfiler` | Rango de fechas, patrones temporales, detección de componente de hora |
| `DatasetProfiler` | Orquestación completa: resumen, perfiles por columna, análisis de nulos, puntuación de calidad |

## Puntuación de Calidad

La puntuación global de calidad (0–100) y calificación de letra (A/B/C/D) se calcula en 4 dimensiones:

| Dimensión | Descripción |
|---|---|
| Completitud | Penaliza los valores faltantes |
| Unicidad | Penaliza las filas duplicadas |
| Consistencia | Penaliza las columnas constantes/inútiles |
| Validez | Marca columnas potencialmente identificadoras almacenadas como categóricas |

## Uso

```python
from data_profiling import DatasetProfiler

perfilador = DatasetProfiler()
reporte = perfilador.profile(df)

# Imprimir resumen formateado
perfilador.print_summary(reporte)

# Obtener tabla ordenada
print(perfilador.summary_table(reporte))

# Exportar a JSON
perfilador.export_json(reporte, "reporte_perfil.json")
```

## Dependencias

```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0
```
