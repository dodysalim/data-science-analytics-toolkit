# 🤖 Machine Learning Pipeline

Pipeline completo y reutilizable para proyectos de **Machine Learning** con Scikit-Learn. Compara múltiples modelos automáticamente y selecciona el mejor.

## ✨ Características

- 🔄 Soporta **Clasificación** y **Regresión**
- 📊 Compara **8 modelos de clasificación** y **7 de regresión** automáticamente
- 📈 Validación cruzada (K-Fold) para todos los modelos
- 🎯 Métricas completas: Accuracy, F1, Precision, Recall, R², RMSE, MAE
- 📉 Visualizaciones: Ranking, Matriz de Confusión, Feature Importance
- 🔍 Búsqueda de hiperparámetros con GridSearchCV
- 💾 Guardado y carga de modelos con joblib

## 🚀 Uso Rápido

```python
from ml_pipeline import MLPipeline

# 1. Crear pipeline
pipeline = MLPipeline(task='classification', scale=True)

# 2. Preparar datos
pipeline.prepare_data(X, y)

# 3. Comparar modelos
rankings = pipeline.compare_models()

# 4. Visualizar resultados
pipeline.plot_rankings()
pipeline.plot_confusion_matrix()
pipeline.plot_feature_importance(feature_names=X.columns.tolist())

# 5. Guardar mejor modelo
pipeline.save_model('mi_modelo.joblib')
```

## 📊 Modelos incluidos

### Clasificación
`Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`, `Extra Trees`, `SVM`, `KNN`, `Naive Bayes`

### Regresión
`Linear Regression`, `Ridge`, `Lasso`, `ElasticNet`, `Decision Tree`, `Random Forest`, `Gradient Boosting`

## 📦 Instalación

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

---
> **Autor:** Dody Dueñas | Data Analyst & Data Scientist
