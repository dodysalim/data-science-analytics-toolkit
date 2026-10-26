# 📐 Statistical Analysis Toolkit — Análisis Estadístico Avanzado

Toolkit Python para realizar pruebas estadísticas formales, validar supuestos de modelos y comparar grupos de forma rigurosa.

## 📁 Estructura

```
05_statistical_analysis/
├── statistical_analysis.py    # Toolkit principal
└── README.md
```

## 🧪 Análisis Incluidos

| Análisis | Descripción |
|----------|-------------|
| Test de Normalidad (Shapiro-Wilk) | ¿Sigue la variable una distribución normal? |
| Test de Normalidad (KS-Test) | Kolmogorov-Smirnov para muestras grandes |
| Test T de Student | Comparación de medias entre dos grupos |
| ANOVA | Comparación de medias entre tres o más grupos |
| Mann-Whitney U | Alternativa no paramétrica al T-Test |
| Chi-Cuadrado | Independencia entre variables categóricas |
| Correlación (Pearson/Spearman) | Fuerza y dirección de la relación |
| Intervalo de Confianza | Estimación del parámetro poblacional |

## 🚀 Uso

```python
from statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(df)
analyzer.normality_test(column="age")
analyzer.ttest(group_col="survived", value_col="fare")
```
