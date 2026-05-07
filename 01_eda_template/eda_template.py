"""
📊 EDA Template - Exploratory Data Analysis
============================================
Clase reutilizable para realizar EDA profesional en proyectos de Data Science.

Autor: Dody Dueñas
Fecha: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
#  Configuración de estilo global
# ──────────────────────────────────────────────
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style="darkgrid", palette="viridis")


class EDAAnalyzer:
    """
    Clase principal para Análisis Exploratorio de Datos (EDA).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos a analizar.
    target : str, opcional
        Nombre de la columna objetivo (variable dependiente).
    """

    def __init__(self, df: pd.DataFrame, target: str = None):
        self.df = df.copy()
        self.target = target
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"✅ EDAAnalyzer iniciado con {df.shape[0]:,} filas y {df.shape[1]} columnas.")

    # ──────────────────────────────────────────────
    #  1. Resumen General del Dataset
    # ──────────────────────────────────────────────
    def overview(self) -> pd.DataFrame:
        """Genera un resumen completo de todas las columnas del dataset."""
        print("\n" + "="*60)
        print("📋  RESUMEN GENERAL DEL DATASET")
        print("="*60)
        print(f"  Filas:    {self.df.shape[0]:,}")
        print(f"  Columnas: {self.df.shape[1]}")
        print(f"  Memoria:  {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

        summary = pd.DataFrame({
            'Tipo': self.df.dtypes,
            'Nulos': self.df.isnull().sum(),
            'Nulos_%': (self.df.isnull().sum() / len(self.df) * 100).round(2),
            'Únicos': self.df.nunique(),
            'Ejemplo': self.df.iloc[0]
        })
        print(summary.to_string())
        return summary

    # ──────────────────────────────────────────────
    #  2. Estadísticas Descriptivas
    # ──────────────────────────────────────────────
    def descriptive_stats(self) -> pd.DataFrame:
        """Estadísticas descriptivas extendidas para columnas numéricas."""
        print("\n" + "="*60)
        print("📈  ESTADÍSTICAS DESCRIPTIVAS")
        print("="*60)
        desc = self.df[self.numeric_cols].describe().T
        desc['skewness'] = self.df[self.numeric_cols].skew().round(3)
        desc['kurtosis'] = self.df[self.numeric_cols].kurtosis().round(3)
        desc['cv_%'] = (desc['std'] / desc['mean'] * 100).round(2)
        print(desc.to_string())
        return desc

    # ──────────────────────────────────────────────
    #  3. Detección de Outliers
    # ──────────────────────────────────────────────
    def detect_outliers(self) -> pd.DataFrame:
        """
        Detecta outliers usando dos métodos:
        - IQR (Rango Intercuartílico)
        - Z-Score (|z| > 3)
        """
        print("\n" + "="*60)
        print("🚨  DETECCIÓN DE OUTLIERS")
        print("="*60)
        results = []
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()

            # IQR Method
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)).sum()

            # Z-Score Method
            z_scores = np.abs(stats.zscore(col_data))
            zscore_outliers = (z_scores > 3).sum()

            results.append({
                'Columna': col,
                'Outliers_IQR': iqr_outliers,
                'Outliers_IQR_%': round(iqr_outliers / len(col_data) * 100, 2),
                'Outliers_ZScore': zscore_outliers,
                'Outliers_ZScore_%': round(zscore_outliers / len(col_data) * 100, 2)
            })

        result_df = pd.DataFrame(results).set_index('Columna')
        print(result_df.to_string())
        return result_df

    # ──────────────────────────────────────────────
    #  4. Visualización: Correlaciones
    # ──────────────────────────────────────────────
    def plot_correlations(self, method: str = 'pearson'):
        """
        Genera un heatmap de correlaciones entre variables numéricas.

        Parámetros
        ----------
        method : str
            Método de correlación: 'pearson', 'spearman' o 'kendall'.
        """
        print(f"\n📊 Generando Heatmap de Correlaciones ({method.capitalize()})...")
        corr = self.df[self.numeric_cols].corr(method=method)

        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax,
            annot_kws={'size': 9}
        )
        ax.set_title(f'Matriz de Correlaciones ({method.capitalize()})', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  ✅ Guardado: correlation_heatmap.png")

    # ──────────────────────────────────────────────
    #  5. Visualización: Distribuciones
    # ──────────────────────────────────────────────
    def plot_distributions(self, cols: list = None, bins: int = 30):
        """
        Grafica la distribución de todas las columnas numéricas.

        Parámetros
        ----------
        cols : list, opcional
            Lista de columnas específicas. Si None, usa todas las numéricas.
        bins : int
            Número de bins para el histograma.
        """
        cols = cols or self.numeric_cols
        n = len(cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            data = self.df[col].dropna()
            axes[i].hist(data, bins=bins, color='steelblue', edgecolor='white', alpha=0.8)
            axes[i].axvline(data.mean(), color='red', linestyle='--', label=f'Media: {data.mean():.2f}')
            axes[i].axvline(data.median(), color='green', linestyle='--', label=f'Mediana: {data.median():.2f}')
            axes[i].set_title(col, fontsize=12)
            axes[i].legend(fontsize=9)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle('Distribuciones de Variables Numéricas', fontsize=18, y=1.01)
        plt.tight_layout()
        plt.savefig('distributions.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  ✅ Guardado: distributions.png")

    # ──────────────────────────────────────────────
    #  6. Análisis de Variables Categóricas
    # ──────────────────────────────────────────────
    def analyze_categoricals(self, top_n: int = 10):
        """
        Visualiza la frecuencia de las categorías más comunes.

        Parámetros
        ----------
        top_n : int
            Número máximo de categorías a mostrar por variable.
        """
        if not self.categorical_cols:
            print("⚠️  No se encontraron columnas categóricas.")
            return

        n = len(self.categorical_cols)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
        axes = axes.flatten()

        for i, col in enumerate(self.categorical_cols):
            top = self.df[col].value_counts().nlargest(top_n)
            axes[i].barh(top.index.astype(str), top.values, color=sns.color_palette("viridis", len(top)))
            axes[i].set_title(f'{col} (Top {top_n})', fontsize=12)
            axes[i].invert_yaxis()

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle('Frecuencia de Variables Categóricas', fontsize=18, y=1.01)
        plt.tight_layout()
        plt.savefig('categoricals.png', dpi=150, bbox_inches='tight')
        plt.show()

    # ──────────────────────────────────────────────
    #  7. EDA Completo (Pipeline)
    # ──────────────────────────────────────────────
    def run_full_analysis(self):
        """
        Ejecuta el pipeline completo de EDA:
        1. Resumen general
        2. Estadísticas descriptivas
        3. Detección de outliers
        4. Heatmap de correlaciones
        5. Distribuciones
        6. Variables categóricas
        """
        print("\n🚀 Iniciando EDA Completo...")
        print("=" * 60)
        self.overview()
        self.descriptive_stats()
        self.detect_outliers()
        if len(self.numeric_cols) > 1:
            self.plot_correlations()
        self.plot_distributions()
        self.analyze_categoricals()
        print("\n✅ EDA Completo finalizado exitosamente.")


# ──────────────────────────────────────────────
#  Ejemplo de uso
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species'] = pd.Categorical(df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'}))

    analyzer = EDAAnalyzer(df, target='target')
    analyzer.run_full_analysis()
