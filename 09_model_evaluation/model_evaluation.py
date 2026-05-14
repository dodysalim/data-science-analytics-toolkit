"""
Kit de Evaluación de Modelos e Informes
=========================================
Suite completa de evaluación para modelos de clasificación y regresión:
- Validación cruzada con múltiples métricas
- Análisis de matriz de confusión
- Cálculo de curvas ROC y PR
- Evaluación de calibración
- Diagnóstico de errores de regresión
- Generación automática de informes HTML/texto

Autor: Dody Dueñas
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
# EVALUADOR DE CLASIFICACIÓN
# ──────────────────────────────────────────────────────────────────────────────

class ClassificationEvaluator:
    """
    Suite completa de evaluación para clasificadores binarios y multiclase.

    Métodos
    -------
    evaluate(y_true, y_pred, y_proba)
        Calcula todas las métricas de clasificación.
    cross_validate_model(modelo, X, y, cv, scoring)
        Ejecuta validación cruzada estratificada k-fold.
    confusion_matrix_analysis(y_true, y_pred)
        Matriz de confusión detallada con precisión, recall y F1 por clase.
    roc_analysis(y_true, y_proba)
        Datos de la curva ROC (solo binario).
    pr_analysis(y_true, y_proba)
        Datos de la curva Precisión-Recall (solo binario).
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        promedio: str = "weighted",
    ) -> Dict[str, float]:
        """
        Calcula métricas de clasificación.

        Parámetros
        ----------
        y_true : array-like
            Etiquetas reales.
        y_pred : array-like
            Etiquetas predichas.
        y_proba : array-like, opcional
            Probabilidades predichas (necesarias para ROC-AUC).
        promedio : str
            Estrategia de promedio para multiclase ('weighted', 'macro', 'micro').

        Retorna
        -------
        dict
            Diccionario de nombre de métrica → valor.
        """
        metricas: Dict[str, float] = {
            "exactitud": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=promedio, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=promedio, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=promedio, zero_division=0),
        }
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    proba_1d = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    metricas["roc_auc"] = roc_auc_score(y_true, proba_1d)
                    metricas["precision_promedio"] = average_precision_score(y_true, proba_1d)
                else:
                    metricas["roc_auc_ovr"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average=promedio
                    )
            except Exception:
                pass
        return metricas

    def cross_validate_model(
        self,
        modelo: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Optional[List[str]] = None,
        estratificado: bool = True,
    ) -> pd.DataFrame:
        """
        Ejecuta validación cruzada y retorna métricas por fold.

        Parámetros
        ----------
        modelo : estimador sklearn
            Cualquier modelo compatible con sklearn (ajustado o no).
        X : array-like
            Matriz de características.
        y : array-like
            Vector objetivo.
        cv : int
            Número de folds.
        scoring : lista de str, opcional
            Nombres de métricas de sklearn. Por defecto: exactitud, f1_weighted, roc_auc.
        estratificado : bool
            Usar StratifiedKFold (True) o KFold (False).

        Retorna
        -------
        pd.DataFrame
            Media ± desv. estándar de cada métrica por fold.
        """
        if scoring is None:
            scoring = ["accuracy", "f1_weighted", "roc_auc"]
        divisor = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if estratificado \
            else KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_resultados = cross_validate(modelo, X, y, cv=divisor, scoring=scoring, return_train_score=True)
        resumen = {}
        for clave, valores in cv_resultados.items():
            if clave.startswith("test_") or clave.startswith("train_"):
                resumen[f"{clave}_media"] = valores.mean()
                resumen[f"{clave}_std"] = valores.std()
        return pd.Series(resumen).to_frame("valor").round(4)

    def confusion_matrix_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        etiquetas: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Análisis detallado de la matriz de confusión.

        Retorna
        -------
        dict con claves: 'matrix' (pd.DataFrame), 'report' (str), 'normalized' (pd.DataFrame)
        """
        cm = confusion_matrix(y_true, y_pred)
        nombres_etiquetas = etiquetas or sorted(np.unique(y_true).tolist())
        cm_df = pd.DataFrame(cm, index=nombres_etiquetas, columns=nombres_etiquetas)
        cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0).round(3)
        informe = classification_report(y_true, y_pred, target_names=[str(l) for l in nombres_etiquetas])
        return {"matrix": cm_df, "normalized": cm_norm, "report": informe}

    def roc_analysis(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calcula los datos de la curva ROC para clasificación binaria.

        Retorna
        -------
        dict con 'fpr', 'tpr', 'umbrales', 'auc'
        """
        proba_1d = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        fpr, tpr, umbrales = roc_curve(y_true, proba_1d)
        auc = roc_auc_score(y_true, proba_1d)
        return {"fpr": fpr, "tpr": tpr, "umbrales": umbrales, "auc": auc}

    def pr_analysis(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calcula los datos de la curva Precisión-Recall para clasificación binaria.

        Retorna
        -------
        dict con 'precision', 'recall', 'umbrales', 'precision_promedio'
        """
        proba_1d = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        precision, recall, umbrales = precision_recall_curve(y_true, proba_1d)
        prec_prom = average_precision_score(y_true, proba_1d)
        return {
            "precision": precision,
            "recall": recall,
            "umbrales": umbrales,
            "precision_promedio": prec_prom,
        }

    def calibration_analysis(
        self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Calcula la curva de calibración (fracción de positivos vs probabilidad media predicha).

        Retorna
        -------
        pd.DataFrame con columnas: prob_predicha_media, fraccion_positivos
        """
        proba_1d = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        fraccion_pos, media_pred = calibration_curve(y_true, proba_1d, n_bins=n_bins)
        return pd.DataFrame({
            "prob_predicha_media": media_pred,
            "fraccion_positivos": fraccion_pos,
        })


# ──────────────────────────────────────────────────────────────────────────────
# EVALUADOR DE REGRESIÓN
# ──────────────────────────────────────────────────────────────────────────────

class RegressionEvaluator:
    """
    Suite completa de evaluación para modelos de regresión.

    Métricas: MAE, MSE, RMSE, MAPE, R², R² ajustado, diagnóstico de residuos.
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_caracteristicas: int = 1,
    ) -> Dict[str, float]:
        """
        Calcula métricas de regresión.

        Parámetros
        ----------
        y_true : array-like
            Valores reales.
        y_pred : array-like
            Valores predichos.
        n_caracteristicas : int
            Número de características (necesario para R² ajustado).

        Retorna
        -------
        dict de métrica → valor
        """
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        r2_ajust = 1 - (1 - r2) * (n - 1) / (n - n_caracteristicas - 1) if n > n_caracteristicas + 1 else np.nan
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "R2": r2,
            "R2_ajustado": r2_ajust,
        }

    def analisis_residuos(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Calcula los residuos y diagnósticos.

        Retorna
        -------
        pd.DataFrame con: real, predicho, residuo, error_abs, error_pct
        """
        residuos = y_true - y_pred
        return pd.DataFrame({
            "real": y_true,
            "predicho": y_pred,
            "residuo": residuos,
            "error_abs": np.abs(residuos),
            "error_pct": np.abs(residuos / np.where(y_true == 0, 1e-10, y_true)) * 100,
        })

    def cross_validate_model(
        self,
        modelo: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Ejecuta validación cruzada k-fold para modelos de regresión."""
        if scoring is None:
            scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
        cv_resultados = cross_validate(
            modelo, X, y,
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring,
            return_train_score=True,
        )
        resumen = {}
        for clave, valores in cv_resultados.items():
            if clave.startswith("test_") or clave.startswith("train_"):
                resumen[f"{clave}_media"] = valores.mean()
                resumen[f"{clave}_std"] = valores.std()
        return pd.Series(resumen).to_frame("valor").round(4)


# ──────────────────────────────────────────────────────────────────────────────
# COMPARADOR DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────

class ModelComparator:
    """
    Compara múltiples modelos mediante validación cruzada y genera una tabla clasificatoria.
    """

    def __init__(self, task: str = "classification", cv: int = 5):
        self.task = task
        self.cv = cv
        self._resultados: Dict[str, pd.Series] = {}

    def add_model(
        self, nombre: str, modelo: Any, X: np.ndarray, y: np.ndarray
    ) -> "ModelComparator":
        """
        Agrega un modelo a la comparación.

        Parámetros
        ----------
        nombre : str
            Nombre de visualización del modelo.
        modelo : estimador sklearn
        X, y : datos de entrenamiento

        Retorna
        -------
        self (para encadenamiento de métodos)
        """
        if self.task == "classification":
            evaluador = ClassificationEvaluator()
            cv_df = evaluador.cross_validate_model(modelo, X, y, cv=self.cv)
        else:
            evaluador = RegressionEvaluator()
            cv_df = evaluador.cross_validate_model(modelo, X, y, cv=self.cv)
        self._resultados[nombre] = cv_df["valor"]
        return self

    def leaderboard(self) -> pd.DataFrame:
        """Retorna una tabla comparativa ordenada de todos los modelos agregados."""
        if not self._resultados:
            raise ValueError("No se agregaron modelos. Llame a add_model() primero.")
        tabla = pd.DataFrame(self._resultados).T
        col_orden = (
            "test_roc_auc_media" if "test_roc_auc_media" in tabla.columns
            else "test_r2_media" if "test_r2_media" in tabla.columns
            else tabla.columns[0]
        )
        return tabla.sort_values(col_orden, ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# GENERADOR DE INFORMES
# ──────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """Genera informes de evaluación en texto plano o Markdown."""

    def classification_report_md(
        self,
        nombre_modelo: str,
        metricas: Dict[str, float],
        analisis_cm: Dict[str, Any],
    ) -> str:
        """
        Genera un informe de clasificación en formato Markdown.

        Parámetros
        ----------
        nombre_modelo : str
        metricas : dict
            Salida de ClassificationEvaluator.evaluate()
        analisis_cm : dict
            Salida de ClassificationEvaluator.confusion_matrix_analysis()

        Retorna
        -------
        str — Informe Markdown
        """
        lineas = [
            f"# Informe de Evaluación: {nombre_modelo}",
            "",
            "## Métricas de Desempeño",
            "",
            "| Métrica | Valor |",
            "|---------|-------|",
        ]
        for metrica, valor in metricas.items():
            lineas.append(f"| {metrica} | {valor:.4f} |")
        lineas += [
            "",
            "## Informe de Clasificación",
            "",
            "```",
            analisis_cm["report"],
            "```",
            "",
            "## Matriz de Confusión",
            "",
            analisis_cm["matrix"].to_markdown() if hasattr(analisis_cm["matrix"], "to_markdown")
            else str(analisis_cm["matrix"]),
        ]
        return "\n".join(lineas)

    def regression_report_md(
        self, nombre_modelo: str, metricas: Dict[str, float]
    ) -> str:
        """Genera un informe de regresión en formato Markdown."""
        lineas = [
            f"# Informe de Evaluación de Regresión: {nombre_modelo}",
            "",
            "## Métricas de Desempeño",
            "",
            "| Métrica | Valor |",
            "|---------|-------|",
        ]
        for metrica, valor in metricas.items():
            lineas.append(f"| {metrica} | {valor:.6f} |")
        return "\n".join(lineas)


# ──────────────────────────────────────────────────────────────────────────────
# Demostración
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

    evaluador = ClassificationEvaluator()
    metricas = evaluador.evaluate(y_test, y_pred, y_proba)
    analisis_cm = evaluador.confusion_matrix_analysis(y_test, y_pred)

    print("=== Métricas ===")
    for k, v in metricas.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Comparación de Modelos ===")
    comparador = ModelComparator(task="classification", cv=5)
    comparador.add_model("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42), X, y)
    comparador.add_model("GradientBoosting", GradientBoostingClassifier(random_state=42), X, y)
    comparador.add_model("RegresionLogistica", LogisticRegression(max_iter=1000, random_state=42), X, y)
    print(comparador.leaderboard())

    generador = ReportGenerator()
    informe_md = generador.classification_report_md("RandomForest", metricas, analisis_cm)
    print("\n=== Vista Previa del Informe Markdown ===")
    print(informe_md[:600])
