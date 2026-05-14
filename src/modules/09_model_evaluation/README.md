# 09 — Kit de Evaluación de Modelos

Suite completa de evaluación para modelos de clasificación y regresión: validación cruzada, curvas ROC/PR, análisis de calibración y generación automática de informes en Markdown.

## Autor

**Dody Dueñas**

## Clases

| Clase | Descripción |
|---|---|
| `ClassificationEvaluator` | Exactitud, Precisión, Recall, F1, ROC-AUC, Precisión Promedio, matriz de confusión, calibración |
| `RegressionEvaluator` | MAE, MSE, RMSE, MAPE, R², R² ajustado, análisis de residuos |
| `ModelComparator` | Compara múltiples modelos mediante tabla de clasificación con validación cruzada |
| `ReportGenerator` | Genera automáticamente informes de evaluación en Markdown |

## Uso

```python
from model_evaluation import ClassificationEvaluator, ModelComparator

evaluador = ClassificationEvaluator()
metricas = evaluador.evaluate(y_test, y_pred, y_proba)
mc = evaluador.confusion_matrix_analysis(y_test, y_pred)
roc = evaluador.roc_analysis(y_test, y_proba)

# Comparar modelos
comparador = ModelComparator(task="classification", cv=5)
comparador.add_model("RF", modelo_rf, X, y)
comparador.add_model("GBT", modelo_gbt, X, y)
print(comparador.leaderboard())
```

## Dependencias

```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
```
