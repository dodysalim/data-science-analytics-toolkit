# 📐 Statistical Analysis Toolkit

Herramientas completas para **análisis estadístico descriptivo e inferencial** en Python. Ideal para proyectos de Data Science y reportes analíticos.

## ✨ Funcionalidades

### 📊 Estadística Descriptiva
- `extended_describe()` → Media, mediana, moda, CV, skewness, kurtosis, IQR

### 🔬 Pruebas de Normalidad
- `test_normality()` → Shapiro-Wilk, Kolmogorov-Smirnov, D'Agostino-Pearson con consenso automático

### 🧪 Pruebas de Hipótesis
| Función | Prueba | Tipo |
|---------|--------|------|
| `one_sample_ttest()` | T-Test una muestra | Paramétrico |
| `two_sample_ttest()` | T-Test dos muestras (Welch/Student) | Paramétrico |
| `anova_test()` | ANOVA de una vía | Paramétrico |
| `chi2_test()` | Chi-cuadrado de independencia | No paramétrico |
| `mann_whitney_test()` | Mann-Whitney U | No paramétrico |

### 📏 Intervalos de Confianza
- `confidence_interval()` → IC para la media con T de Student

### 🔗 Correlación con Significancia
- `correlation_analysis()` → Pearson, Spearman, Kendall con p-values

### 🎨 Visualizaciones
- `plot_normality_check()` → Histograma + Q-Q Plot + Boxplot + Violin
- `plot_hypothesis_comparison()` → Comparación visual de dos grupos

## 🚀 Uso Rápido

```python
from statistical_analysis import *

# Estadísticas descriptivas
extended_describe(df['ventas'])

# Prueba de normalidad
test_normality(df['ventas'])

# Comparar dos grupos
two_sample_ttest(grupo_A, grupo_B, alpha=0.05)

# Visualizar normalidad
plot_normality_check(df['ventas'])
```

## 📦 Instalación

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

---
> **Autor:** Dody Dueñas | Data Analyst & Data Scientist
