"""
🤖 Machine Learning Pipeline
==============================
Pipeline completo para entrenamiento, evaluación y comparación de modelos
de Machine Learning usando Scikit-Learn.

Incluye:
- Clasificación y Regresión
- Comparación automática de múltiples modelos
- Evaluación con métricas detalladas
- Validación cruzada
- Búsqueda de hiperparámetros (GridSearchCV)
- Guardado y carga de modelos

Autor: Dody Dueñas
Fecha: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Modelos de Clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Modelos de Regresión
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ══════════════════════════════════════════════════════════════
#  CONFIGURACIONES DE MODELOS
# ══════════════════════════════════════════════════════════════

CLASSIFICATION_MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
}

REGRESSION_MODELS = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}


# ══════════════════════════════════════════════════════════════
#  CLASE PRINCIPAL: MLPipeline
# ══════════════════════════════════════════════════════════════

class MLPipeline:
    """
    Pipeline completo de Machine Learning para clasificación y regresión.

    Parámetros
    ----------
    task : str
        'classification' o 'regression'
    test_size : float
        Proporción de datos para test (default: 0.2)
    random_state : int
        Semilla aleatoria para reproducibilidad
    scale : bool
        Si True, aplica StandardScaler antes del modelado
    """

    def __init__(
        self,
        task: str = 'classification',
        test_size: float = 0.2,
        random_state: int = 42,
        scale: bool = True
    ):
        if task not in ('classification', 'regression'):
            raise ValueError("task debe ser 'classification' o 'regression'")
        self.task = task
        self.test_size = test_size
        self.random_state = random_state
        self.scale = scale
        self.results_ = {}
        self.best_model_ = None
        self.best_model_name_ = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler() if scale else None

    # ──────────────────────────────────────────────
    #  Preparar datos
    # ──────────────────────────────────────────────
    def prepare_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Divide y (opcionalmente) escala los datos.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if self.task == 'classification' else None
        )
        if self.scale:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

        print(f"✅ Datos preparados:")
        print(f"   Train: {len(self.y_train):,} muestras | Test: {len(self.y_test):,} muestras")
        return self

    # ──────────────────────────────────────────────
    #  Evaluar un solo modelo
    # ──────────────────────────────────────────────
    def evaluate_model(self, model, name: str) -> dict:
        """
        Entrena y evalúa un modelo. Retorna un diccionario con métricas.
        """
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        if self.task == 'classification':
            metrics = {
                'model': model,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred, average='weighted'),
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted'),
            }
            # Validación cruzada
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
        else:
            metrics = {
                'model': model,
                'r2': r2_score(self.y_test, y_pred),
                'mae': mean_absolute_error(self.y_test, y_pred),
                'mse': mean_squared_error(self.y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            }
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()

        return metrics

    # ──────────────────────────────────────────────
    #  Comparar múltiples modelos
    # ──────────────────────────────────────────────
    def compare_models(self, custom_models: dict = None) -> pd.DataFrame:
        """
        Entrena y compara múltiples modelos, retornando un ranking.

        Parámetros
        ----------
        custom_models : dict, opcional
            Diccionario {nombre: modelo}. Si None, usa los modelos por defecto.
        """
        models = custom_models or (
            CLASSIFICATION_MODELS if self.task == 'classification' else REGRESSION_MODELS
        )

        print(f"\n🏁 Comparando {len(models)} modelos ({self.task})...\n")
        for name, model in models.items():
            try:
                metrics = self.evaluate_model(model, name)
                self.results_[name] = metrics
                if self.task == 'classification':
                    print(f"  ✅ {name:<25} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | CV: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
                else:
                    print(f"  ✅ {name:<25} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f} | CV: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
            except Exception as e:
                print(f"  ❌ {name}: Error - {e}")

        return self.get_rankings()

    # ──────────────────────────────────────────────
    #  Obtener ranking
    # ──────────────────────────────────────────────
    def get_rankings(self) -> pd.DataFrame:
        """
        Retorna un DataFrame con el ranking de todos los modelos evaluados.
        """
        rows = []
        for name, metrics in self.results_.items():
            row = {'Model': name}
            row.update({k: v for k, v in metrics.items() if k != 'model'})
            rows.append(row)

        df = pd.DataFrame(rows).set_index('Model')
        sort_col = 'accuracy' if self.task == 'classification' else 'r2'
        df = df.sort_values(sort_col, ascending=False)

        # Identificar el mejor modelo
        self.best_model_name_ = df.index[0]
        self.best_model_ = self.results_[self.best_model_name_]['model']

        print(f"\n🏆 Mejor modelo: {self.best_model_name_}")
        return df

    # ──────────────────────────────────────────────
    #  Visualizar ranking
    # ──────────────────────────────────────────────
    def plot_rankings(self):
        """
        Genera gráficos de comparación de modelos.
        """
        df = self.get_rankings()
        metric = 'accuracy' if self.task == 'classification' else 'r2'
        metric_label = 'Accuracy' if self.task == 'classification' else 'R² Score'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Gráfico 1: Métrica principal
        colors = sns.color_palette("viridis", len(df))
        bars = ax1.barh(df.index[::-1], df[metric][::-1], color=colors)
        ax1.set_xlabel(metric_label, fontsize=12)
        ax1.set_title(f'Comparación de Modelos por {metric_label}', fontsize=14)
        for bar, val in zip(bars, df[metric][::-1]):
            ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=10)

        # Gráfico 2: Validación Cruzada
        ax2.barh(df.index[::-1], df['cv_mean'][::-1], xerr=df['cv_std'][::-1],
                 color=colors, capsize=5, alpha=0.85)
        ax2.set_xlabel(f'{metric_label} (CV 5-fold)', fontsize=12)
        ax2.set_title('Validación Cruzada (Media ± Std)', fontsize=14)

        plt.suptitle(f'Comparación de Modelos - {self.task.capitalize()}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    # ──────────────────────────────────────────────
    #  Matriz de Confusión (solo clasificación)
    # ──────────────────────────────────────────────
    def plot_confusion_matrix(self, model_name: str = None):
        """
        Grafica la matriz de confusión del mejor modelo (o uno específico).
        """
        if self.task != 'classification':
            print("⚠️  La matriz de confusión solo aplica a clasificación.")
            return

        name = model_name or self.best_model_name_
        model = self.results_[name]['model']
        y_pred = model.predict(self.X_test)

        cm = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, ax=ax)
        ax.set_xlabel('Predicción', fontsize=12)
        ax.set_ylabel('Real', fontsize=12)
        ax.set_title(f'Matriz de Confusión - {name}', fontsize=14)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n📊 Classification Report - {name}:")
        print(classification_report(self.y_test, y_pred))

    # ──────────────────────────────────────────────
    #  Feature Importance
    # ──────────────────────────────────────────────
    def plot_feature_importance(self, feature_names: list, model_name: str = None, top_n: int = 15):
        """
        Grafica la importancia de características del mejor modelo.
        """
        name = model_name or self.best_model_name_
        model = self.results_[name]['model']

        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️  '{name}' no soporta feature_importances_.")
            return

        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances = importances.nlargest(top_n).sort_values()

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("viridis", len(importances))
        ax.barh(importances.index, importances.values, color=colors)
        ax.set_xlabel('Importancia', fontsize=12)
        ax.set_title(f'Top {top_n} Features - {name}', fontsize=14)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()

    # ──────────────────────────────────────────────
    #  Guardar / Cargar Modelos
    # ──────────────────────────────────────────────
    def save_model(self, path: str = 'best_model.joblib', model_name: str = None):
        """Guarda el mejor modelo (o uno específico) en disco."""
        name = model_name or self.best_model_name_
        model = self.results_[name]['model']
        joblib.dump({'model': model, 'scaler': self.scaler, 'name': name}, path)
        print(f"💾 Modelo '{name}' guardado en: {path}")

    @staticmethod
    def load_model(path: str):
        """Carga un modelo guardado desde disco."""
        data = joblib.load(path)
        print(f"✅ Modelo '{data['name']}' cargado desde: {path}")
        return data


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN HELPER: Búsqueda de Hiperparámetros
# ══════════════════════════════════════════════════════════════

def hyperparameter_tuning(
    model,
    param_grid: dict,
    X_train, y_train,
    scoring: str = 'accuracy',
    cv: int = 5,
    n_jobs: int = -1
) -> GridSearchCV:
    """
    Realiza búsqueda de hiperparámetros con GridSearchCV.

    Returns
    -------
    GridSearchCV ajustado con los mejores parámetros.
    """
    print(f"🔍 Iniciando GridSearchCV con {len(param_grid)} parámetros...")
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=1)
    grid.fit(X_train, y_train)
    print(f"✅ Mejores parámetros: {grid.best_params_}")
    print(f"   Mejor score ({scoring}): {grid.best_score_:.4f}")
    return grid


# ──────────────────────────────────────────────
#  Ejemplo de uso: Clasificación con Iris
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    print("="*60)
    print("🤖  ML PIPELINE - Clasificación con Iris Dataset")
    print("="*60)

    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    pipeline = MLPipeline(task='classification', test_size=0.2, scale=True)
    pipeline.prepare_data(X, y)
    rankings = pipeline.compare_models()

    print("\n📊 Rankings:")
    print(rankings[['accuracy', 'f1', 'cv_mean', 'cv_std']].to_string())

    pipeline.plot_rankings()
    pipeline.plot_confusion_matrix()
    pipeline.plot_feature_importance(feature_names=iris.feature_names)
    pipeline.save_model('iris_best_model.joblib')
