# 📊 Exploratory Data Analysis (EDA) Template

Un template profesional y reutilizable para realizar Análisis Exploratorio de Datos (EDA) en proyectos de Data Science.

## 🚀 Características

- ✅ Análisis automático de tipos de datos y valores nulos
- ✅ Visualizaciones estadísticas descriptivas
- ✅ Detección de outliers con IQR y Z-Score
- ✅ Análisis de correlaciones con heatmaps
- ✅ Distribuciones univariadas y bivariadas
- ✅ Informe de calidad de datos

## 📁 Estructura

```
01_eda_template/
├── eda_template.py       # Clase principal de EDA
├── eda_notebook.ipynb    # Notebook interactivo
├── requirements.txt      # Dependencias
└── README.md
```

## 🛠️ Instalación

```bash
pip install -r requirements.txt
```

## 📌 Uso

```python
from eda_template import EDAAnalyzer

analyzer = EDAAnalyzer(df)
analyzer.run_full_analysis()
```

## 🧰 Tecnologías

- Python 3.9+
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scipy

---
> **Autor:** Dody Dueñas | Data Analyst & Data Scientist
