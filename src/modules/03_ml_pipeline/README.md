# 🤖 ML Pipeline — Pipeline Automatizado de Machine Learning

Pipeline que entrena, evalúa y compara múltiples algoritmos de Machine Learning de forma automática con una sola llamada.

## 📁 Estructura

```
03_ml_pipeline/
├── ml_pipeline.py    # Clase principal del pipeline
└── README.md
```

## 🏆 Modelos Comparados

| Modelo | Tipo |
|--------|------|
| Logistic Regression | Clasificación lineal |
| Decision Tree | Árbol de decisión |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble (boosting) |
| SVM | Support Vector Machine |
| K-Nearest Neighbors | Distancia |
| Naive Bayes | Probabilístico |
| XGBoost | Extreme Gradient Boosting |

## ⚙️ Métricas Evaluadas

`Accuracy` · `Precision` · `Recall` · `F1-Score` · `ROC-AUC` · `Cross-Validation`

## 🚀 Uso

```python
from ml_pipeline import MLPipeline

pipeline = MLPipeline(X_train, X_test, y_train, y_test)
results = pipeline.compare_all_models()
pipeline.plot_results()
```
