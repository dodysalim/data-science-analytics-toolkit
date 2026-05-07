# 📊 EDA Template — Plantilla Maestra de Análisis Exploratorio

Notebook interactivo de **18 pasos** para realizar un Análisis Exploratorio de Datos (EDA) completo y profesional sobre cualquier dataset tabular.

## 📁 Estructura

```
01_eda_template/
├── eda_notebook.ipynb    # Notebook interactivo de 18 pasos (self-contained)
├── requirements.txt      # Dependencias
└── README.md
```

## 🔬 Los 18 Pasos del EDA

| # | Paso | Descripción |
|---|------|-------------|
| 1 | Configuración | Importación de librerías y estilo global |
| 2 | Carga del Dataset | CSV, Parquet, o dataset de ejemplo |
| 3 | Snapshot | Dimensiones, head y muestra aleatoria |
| 4 | Metadata | Tipos de datos y estructura |
| 5 | Diccionario de Variables | Separa numéricas de categóricas |
| 6 | Valores Nulos | Reporte de nulos y proporción |
| 7 | Tratamiento de Nulos | Imputación con mediana / moda |
| 8 | Duplicados | Detección y eliminación |
| 9 | Estadísticas Numéricas | Media, mediana, desviación estándar |
| 10 | Estadísticas Categóricas | Frecuencias y cardinalidad |
| 11 | Distribuciones Numéricas | Histogramas + KDE |
| 12 | Distribuciones Categóricas | Barplots de frecuencia |
| 13 | Outliers (IQR) | Detección matemática via Rango Intercuartílico |
| 14 | Outliers (Visual) | Boxplots multivariados |
| 15 | Correlación | Heatmap de correlación de Pearson |
| 16 | Bivariado | Numéricas vs variable objetivo (violinplots) |
| 17 | Feature Engineering | Creación de nuevas variables derivadas |
| 18 | Conclusiones | Resumen de insights y siguientes pasos |

## 🚀 Uso

```bash
pip install -r requirements.txt
jupyter notebook eda_notebook.ipynb
```
