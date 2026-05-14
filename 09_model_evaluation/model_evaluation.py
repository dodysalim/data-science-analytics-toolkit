"""
Model Evaluation & Reporting Toolkit
======================================
Comprehensive evaluation suite for classification and regression models:
- Cross-validation with multiple metrics
- Confusion matrix analysis
- ROC/PR curve computation
- Calibration assessment
- Regression error diagnostics
- Automated HTML/text report generation

Author: Data Science Analytics Toolkit
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION EVALUATOR
# ──────────────────────────────────────────────────────────────────────────────

class ClassificationEvaluator:
    """
    Full evaluation suite for binary and multi-class classifiers.

    Methods
    -------
    evaluate(y_true, y_pred, y_proba)
        Compute all classification metrics.
    cross_validate_model(model, X, y, cv, scoring)
        Run stratified k-fold cross-validation.
    confusion_matrix_analysis(y_true, y_pred)
        Detailed confusion matrix with per-class precision, recall, F1.
    roc_analysis(y_true, y_proba)
        ROC curve data (binary only).
    pr_analysis(y_true, y_proba)
        Precision-recall curve data (binary only).
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = "weighted",
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.
        y_pred : array-like
            Predicted labels.
        y_proba : array-like, optional
            Predicted probabilities (needed for ROC-AUC).
        average : str
            Averaging strategy for multi-class ('weighted', 'macro', 'micro').

        Returns
        -------
        dict
            Dictionary of metric name → value.
        """
        metrics: Dict[str, float] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        }
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    proba_1d = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    metrics["roc_auc"] = roc_auc_score(y_true, proba_1d)
                    metrics["avg_precision"] = average_precision_score(y_true, proba_1d)
                else:
                    metrics["roc_auc_ovr"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average=average
                    )
            except Exception:
                pass
        return metrics

    def cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Optional[List[str]] = None,
        stratified: bool = True,
    ) -> pd.DataFrame:
        """
        Run cross-validation and return per-fold metrics.

        Parameters
        ----------
        model : sklearn estimator
            Any fitted or unfitted sklearn-compatible model.
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        cv : int
            Number of folds.
        scoring : list of str, optional
            Sklearn scoring names. Defaults to accuracy, f1_weighted, roc_auc.
        stratified : bool
            Use StratifiedKFold (True) or KFold (False).

        Returns
        -------
        pd.DataFrame
            Mean ± std for each metric across folds.
        """
        if scoring is None:
            scoring = ["accuracy", "f1_weighted", "roc_auc"]
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if stratified \
            else KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X, y, cv=splitter, scoring=scoring, return_train_score=True)
        summary = {}
        for key, values in cv_results.items():
            if key.startswith("test_") or key.startswith("train_"):
                summary[f"{key}_mean"] = values.mean()
                summary[f"{key}_std"] = values.std()
        return pd.Series(summary).to_frame("value").round(4)

    def confusion_matrix_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detailed confusion matrix analysis.

        Returns
        -------
        dict with keys: 'matrix' (pd.DataFrame), 'report' (str), 'normalized' (pd.DataFrame)
        """
        cm = confusion_matrix(y_true, y_pred)
        label_names = labels or sorted(np.unique(y_true).tolist())
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0).round(3)
        report = classification_report(y_true, y_pred, target_names=[str(l) for l in label_names])
        return {"matrix": cm_df, "normalized": cm_norm, "report": report}

    def roc_analysis(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute ROC curve data for binary classification.

        Returns
        -------
        dict with 'fpr', 'tpr', 'thresholds', 'auc'
        """
        proba_1d = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        fpr, tpr, thresholds = roc_curve(y_true, proba_1d)
        auc = roc_auc_score(y_true, proba_1d)
        return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}

    def pr_analysis(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute Precision-Recall curve data for binary classification.

        Returns
        -------
        dict with 'precision', 'recall', 'thresholds', 'avg_precision'
        """
        proba_1d = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        precision, recall, thresholds = precision_recall_curve(y_true, proba_1d)
        avg_prec = average_precision_score(y_true, proba_1d)
        return {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "avg_precision": avg_prec,
        }

    def calibration_analysis(
        self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Compute calibration curve (fraction of positives vs mean predicted probability).

        Returns
        -------
        pd.DataFrame with columns: mean_predicted_prob, fraction_of_positives
        """
        proba_1d = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        fraction_pos, mean_pred = calibration_curve(y_true, proba_1d, n_bins=n_bins)
        return pd.DataFrame({
            "mean_predicted_prob": mean_pred,
            "fraction_of_positives": fraction_pos,
        })


# ──────────────────────────────────────────────────────────────────────────────
# REGRESSION EVALUATOR
# ──────────────────────────────────────────────────────────────────────────────

class RegressionEvaluator:
    """
    Full evaluation suite for regression models.

    Metrics: MAE, MSE, RMSE, MAPE, R², Adjusted R², residual diagnostics.
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: int = 1,
    ) -> Dict[str, float]:
        """
        Compute regression metrics.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.
        y_pred : array-like
            Predicted values.
        n_features : int
            Number of features (needed for adjusted R²).

        Returns
        -------
        dict of metric → value
        """
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else np.nan
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "R2": r2,
            "Adjusted_R2": adj_r2,
        }

    def residual_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute residuals and diagnostics.

        Returns
        -------
        pd.DataFrame with: actual, predicted, residual, abs_error, pct_error
        """
        residuals = y_true - y_pred
        return pd.DataFrame({
            "actual": y_true,
            "predicted": y_pred,
            "residual": residuals,
            "abs_error": np.abs(residuals),
            "pct_error": np.abs(residuals / np.where(y_true == 0, 1e-10, y_true)) * 100,
        })

    def cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run k-fold cross-validation for regression models."""
        if scoring is None:
            scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
        cv_results = cross_validate(
            model, X, y,
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring,
            return_train_score=True,
        )
        summary = {}
        for key, values in cv_results.items():
            if key.startswith("test_") or key.startswith("train_"):
                summary[f"{key}_mean"] = values.mean()
                summary[f"{key}_std"] = values.std()
        return pd.Series(summary).to_frame("value").round(4)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

class ModelComparator:
    """
    Compare multiple models using cross-validation and produce a ranked report.
    """

    def __init__(self, task: str = "classification", cv: int = 5):
        self.task = task
        self.cv = cv
        self._results: Dict[str, pd.Series] = {}

    def add_model(
        self, name: str, model: Any, X: np.ndarray, y: np.ndarray
    ) -> "ModelComparator":
        """
        Add a model to the comparison.

        Parameters
        ----------
        name : str
            Display name for the model.
        model : sklearn estimator
        X, y : training data

        Returns
        -------
        self (for method chaining)
        """
        if self.task == "classification":
            evaluator = ClassificationEvaluator()
            cv_df = evaluator.cross_validate_model(model, X, y, cv=self.cv)
        else:
            evaluator = RegressionEvaluator()
            cv_df = evaluator.cross_validate_model(model, X, y, cv=self.cv)
        self._results[name] = cv_df["value"]
        return self

    def leaderboard(self) -> pd.DataFrame:
        """Return a sorted comparison table of all added models."""
        if not self._results:
            raise ValueError("No models added. Call add_model() first.")
        board = pd.DataFrame(self._results).T
        sort_col = (
            "test_roc_auc_mean" if "test_roc_auc_mean" in board.columns
            else "test_r2_mean" if "test_r2_mean" in board.columns
            else board.columns[0]
        )
        return board.sort_values(sort_col, ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """Generate a plain-text or Markdown evaluation report."""

    def classification_report_md(
        self,
        model_name: str,
        metrics: Dict[str, float],
        cm_analysis: Dict[str, Any],
    ) -> str:
        """
        Generate a Markdown-formatted classification report.

        Parameters
        ----------
        model_name : str
        metrics : dict
            Output from ClassificationEvaluator.evaluate()
        cm_analysis : dict
            Output from ClassificationEvaluator.confusion_matrix_analysis()

        Returns
        -------
        str — Markdown report
        """
        lines = [
            f"# Evaluation Report: {model_name}",
            "",
            "## Performance Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for metric, value in metrics.items():
            lines.append(f"| {metric} | {value:.4f} |")
        lines += [
            "",
            "## Classification Report",
            "",
            "```",
            cm_analysis["report"],
            "```",
            "",
            "## Confusion Matrix",
            "",
            cm_analysis["matrix"].to_markdown() if hasattr(cm_analysis["matrix"], "to_markdown")
            else str(cm_analysis["matrix"]),
        ]
        return "\n".join(lines)

    def regression_report_md(
        self, model_name: str, metrics: Dict[str, float]
    ) -> str:
        """Generate a Markdown-formatted regression report."""
        lines = [
            f"# Regression Evaluation Report: {model_name}",
            "",
            "## Performance Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for metric, value in metrics.items():
            lines.append(f"| {metric} | {value:.6f} |")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    evaluator = ClassificationEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)
    cm_analysis = evaluator.confusion_matrix_analysis(y_test, y_pred)

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Model Comparison ===")
    comparator = ModelComparator(task="classification", cv=5)
    comparator.add_model("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42), X, y)
    comparator.add_model("GradientBoosting", GradientBoostingClassifier(random_state=42), X, y)
    comparator.add_model("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42), X, y)
    print(comparator.leaderboard())

    reporter = ReportGenerator()
    report_md = reporter.classification_report_md("RandomForest", metrics, cm_analysis)
    print("\n=== Markdown Report (preview) ===")
    print(report_md[:600])
