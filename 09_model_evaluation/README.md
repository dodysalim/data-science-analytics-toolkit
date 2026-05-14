# 09 — Model Evaluation & Reporting Toolkit

Full evaluation suite for classification and regression models with cross-validation, ROC/PR curves, calibration analysis, and automated Markdown report generation.

## Classes

| Class | Description |
|---|---|
| `ClassificationEvaluator` | Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision, confusion matrix, calibration |
| `RegressionEvaluator` | MAE, MSE, RMSE, MAPE, R², Adjusted R², residual analysis |
| `ModelComparator` | Compare multiple models via cross-validation leaderboard |
| `ReportGenerator` | Auto-generate Markdown evaluation reports |

## Usage

```python
from model_evaluation import ClassificationEvaluator, ModelComparator

evaluator = ClassificationEvaluator()
metrics = evaluator.evaluate(y_test, y_pred, y_proba)
cm = evaluator.confusion_matrix_analysis(y_test, y_pred)
roc = evaluator.roc_analysis(y_test, y_proba)

# Compare models
comparator = ModelComparator(task="classification", cv=5)
comparator.add_model("RF", rf_model, X, y)
comparator.add_model("GBT", gbt_model, X, y)
print(comparator.leaderboard())
```

## Requirements

```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
```
