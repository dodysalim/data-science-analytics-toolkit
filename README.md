# 📦 Data Science & Analytics Toolkit

Colección de herramientas profesionales de **Data Science** y **Análisis de Datos** construidas en Python, con arquitectura SOLID y código listo para producción.

> **Autor: Dody Dueñas** — Data Analyst & Data Scientist

---

## 🚀 Dashboard Interactivo

El proyecto incluye un **dashboard web interactivo** construido con **Streamlit** que permite explorar todos los módulos visualmente desde el navegador.

```bash
# Iniciar el dashboard
python -m streamlit run dashboard.py
```

Abre **http://localhost:8501** en tu navegador. El dashboard incluye:

| Página | Descripción |
|---|---|
| 🏠 Inicio | Vista general del dataset y métricas clave |
| 🔍 Perfil de Datos | Calificación de calidad A-D, nulos, distribuciones interactivas |
| ⚙️ Ingeniería de Características | Transformaciones en tiempo real con sliders |
| 📈 Series de Tiempo | Pronóstico ARIMA, descomposición y media móvil ajustable |
| 🤖 Evaluación de Modelos | Curva ROC, matriz de confusión, leaderboard de modelos |
| 💬 NLP & Sentimientos | Análisis de texto en vivo + modelado de temas LDA |
| 🧪 Pruebas A/B | Calculadora de potencia + análisis frecuentista y bayesiano |

---

## 🗂️ Estructura del Repositorio

```
data-science-analytics-toolkit/
│
├── dashboard.py                  # Dashboard Streamlit interactivo
├── demo_proyecto_completo.py     # Demo de consola de todos los módulos
│
├── 01_eda_template/              # Plantilla maestra de EDA (18 pasos)
├── 02_data_cleaning_utils/       # Utilidades de limpieza de datos
├── 03_ml_pipeline/               # Pipeline automatizado de Machine Learning
├── 04_sql_analytics/             # Queries SQL avanzadas de negocio
├── 05_statistical_analysis/      # Toolkit de análisis estadístico
├── 06_customer_support_analysis/ # Análisis real con arquitectura SOLID
├── 07_time_series/               # Análisis de series de tiempo
├── 08_feature_engineering/       # Ingeniería de características automatizada
├── 09_model_evaluation/          # Evaluación y comparación de modelos ML
├── 10_data_visualization/        # Gráficos y dashboards profesionales
├── 11_nlp_toolkit/               # Procesamiento de lenguaje natural
├── 12_data_profiling/            # Perfilado y calidad de datos
└── 13_ab_testing/                # Pruebas A/B y análisis de experimentos
```

---

## 📁 Módulos

### 01 · EDA Template
Notebook interactivo de **18 pasos** para el Análisis Exploratorio de cualquier dataset.
→ [Ver carpeta](./01_eda_template)

### 02 · Data Cleaning Utils
Librería reutilizable de funciones de limpieza: manejo de nulos, outliers, encoding y normalización.
→ [Ver carpeta](./02_data_cleaning_utils)

### 03 · ML Pipeline
Pipeline automatizado que entrena y compara múltiples modelos de Machine Learning con una sola llamada.
→ [Ver carpeta](./03_ml_pipeline)

### 04 · SQL Analytics
Colección de queries SQL avanzadas para análisis de negocio: cohorts, RFM, ventanas de tiempo.
→ [Ver carpeta](./04_sql_analytics)

### 05 · Statistical Analysis
Toolkit para pruebas de hipótesis, tests de normalidad y análisis estadístico comparativo.
→ [Ver carpeta](./05_statistical_analysis)

### 06 · Customer Support Analysis ⭐
Análisis completo de **976K conversaciones** de soporte al cliente usando **principios SOLID**.
→ [Ver carpeta](./06_customer_support_analysis)

### 07 · Time Series Analyzer
Pruebas ADF/KPSS, descomposición clásica y STL, modelos ARIMA, Holt-Winters y detección de anomalías.
→ [Ver carpeta](./07_time_series)

### 08 · Feature Engineering Toolkit
Extracción datetime, codificación Target/Frequency, binning, características polinomiales y selección estadística.
→ [Ver carpeta](./08_feature_engineering)

### 09 · Model Evaluation Toolkit
Suite completa: métricas, curvas ROC/PR, calibración, validación cruzada y generación de reportes Markdown.
→ [Ver carpeta](./09_model_evaluation)

### 10 · Data Visualization Toolkit
Gráficos de distribución, correlación, categóricos, series de tiempo y dashboard EDA multipanel.
→ [Ver carpeta](./10_data_visualization)

### 11 · NLP Text Processing Toolkit
Limpieza, TF-IDF, n-gramas, modelado de temas LDA/LSA y análisis de sentimientos por léxico.
→ [Ver carpeta](./11_nlp_toolkit)

### 12 · Data Profiling & Quality Report
Profiler automático con scoring de calidad 0-100 y calificación de letra A-D.
→ [Ver carpeta](./12_data_profiling)

### 13 · A/B Testing & Experiment Analysis
Power analysis, pruebas Z/T/Mann-Whitney/Chi-cuadrado, análisis Bayesiano Beta-Binomial.
→ [Ver carpeta](./13_ab_testing)

---

## 🛠️ Tecnologías

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `SciPy` · `Statsmodels` · `Streamlit` · `Jupyter`

---

## ⚡ Inicio Rápido

```bash
# 1. Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn scipy statsmodels streamlit

# 2. Lanzar el dashboard interactivo
python -m streamlit run dashboard.py

# 3. O correr la demo completa en consola
python -X utf8 demo_proyecto_completo.py
```

---

## 📊 Demostración

Cada módulo tiene su propio bloque `if __name__ == "__main__":` para ejecutarse de forma independiente:

```bash
python 07_time_series/time_series_analyzer.py
python 12_data_profiling/data_profiling.py
python 13_ab_testing/ab_testing.py
```
