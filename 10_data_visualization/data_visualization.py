"""
Kit de Visualización de Datos
================================
Gráficos y dashboards de calidad profesional usando matplotlib y seaborn.
Sigue principios SOLID con clases separadas por dominio de visualización.

Módulos:
  - DistributionPlotter      : histogramas, KDE, Q-Q plots, violin plots
  - CorrelationPlotter       : mapas de calor, pair plots, matrices de dispersión
  - CategoricalPlotter       : gráficos de barras, conteo, gráficos de torta/donut
  - TimeSeriesPlotter        : líneas, promedios móviles, estacionalidad
  - ModelPerformancePlotter  : curvas ROC y PR, matriz de confusión, residuos
  - DashboardBuilder         : composición de figuras multipanel

Autor: Dody Dueñas
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Estilo por defecto ─────────────────────────────────────────────────────────
PALETA = "husl"
ESTILO = "whitegrid"
CONTEXTO = "paper"
DPI = 120

sns.set_theme(style=ESTILO, context=CONTEXTO, palette=PALETA)
plt.rcParams.update({"figure.dpi": DPI, "axes.titlepad": 12})


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICADOR DE DISTRIBUCIONES
# ──────────────────────────────────────────────────────────────────────────────

class DistributionPlotter:
    """Grafica distribuciones univariadas y bivariadas."""

    def histogram_kde(
        self,
        serie: pd.Series,
        bins: int = 30,
        titulo: Optional[str] = None,
        color: str = "#5B84C4",
        figsize: Tuple = (8, 4),
    ) -> plt.Figure:
        """Histograma con curva KDE superpuesta."""
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(serie.dropna(), bins=bins, kde=True, ax=ax, color=color)
        ax.set_title(titulo or f"Distribución de {serie.name or 'Serie'}")
        ax.set_xlabel(serie.name or "Valor")
        ax.set_ylabel("Conteo")
        fig.tight_layout()
        return fig

    def qqplot(
        self,
        serie: pd.Series,
        titulo: Optional[str] = None,
        figsize: Tuple = (6, 5),
    ) -> plt.Figure:
        """Gráfico Q-Q para evaluar normalidad."""
        fig, ax = plt.subplots(figsize=figsize)
        qq = stats.probplot(serie.dropna(), dist="norm")
        cuant_teoricos = qq[0][0]
        cuant_muestra = qq[0][1]
        ax.scatter(cuant_teoricos, cuant_muestra, alpha=0.6, color="#E07B54", s=20)
        ajuste = np.polyfit(cuant_teoricos, cuant_muestra, 1)
        ax.plot(cuant_teoricos, np.polyval(ajuste, cuant_teoricos), color="#333", lw=1.5, ls="--")
        ax.set_title(titulo or f"Gráfico Q-Q: {serie.name or 'Serie'}")
        ax.set_xlabel("Cuantiles Teóricos")
        ax.set_ylabel("Cuantiles Muestrales")
        fig.tight_layout()
        return fig

    def violin_box(
        self,
        df: pd.DataFrame,
        col_numerica: str,
        col_grupo: Optional[str] = None,
        figsize: Tuple = (10, 5),
    ) -> plt.Figure:
        """Gráfico violín + caja opcionalmente agrupado por categoría."""
        fig, ax = plt.subplots(figsize=figsize)
        if col_grupo:
            sns.violinplot(data=df, x=col_grupo, y=col_numerica, ax=ax, inner="box", palette=PALETA)
        else:
            sns.violinplot(data=df, y=col_numerica, ax=ax, inner="box", color="#5B84C4")
        ax.set_title(f"Distribución de {col_numerica}" + (f" por {col_grupo}" if col_grupo else ""))
        fig.tight_layout()
        return fig

    def multi_distribucion(
        self, df: pd.DataFrame, columnas: List[str], figsize: Tuple = (15, 4)
    ) -> plt.Figure:
        """Grafica histogramas para múltiples columnas numéricas en una sola fila."""
        n = len(columnas)
        fig, ejes = plt.subplots(1, n, figsize=figsize)
        ejes = np.atleast_1d(ejes)
        for ax, col in zip(ejes, columnas):
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(col)
            ax.set_xlabel("")
        fig.suptitle("Distribuciones de Variables", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICADOR DE CORRELACIONES
# ──────────────────────────────────────────────────────────────────────────────

class CorrelationPlotter:
    """Visualiza relaciones y correlaciones entre pares de variables."""

    def heatmap(
        self,
        df: pd.DataFrame,
        metodo: str = "pearson",
        figsize: Tuple = (10, 8),
        anotar: bool = True,
    ) -> plt.Figure:
        """
        Mapa de calor de correlaciones.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame numérico.
        metodo : str
            'pearson', 'spearman' o 'kendall'.
        """
        corr = df.select_dtypes(include=np.number).corr(method=metodo)
        mascara = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr, mask=mascara, annot=anotar, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax,
        )
        ax.set_title(f"Matriz de Correlación {metodo.capitalize()}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def pairplot(
        self,
        df: pd.DataFrame,
        hue: Optional[str] = None,
        columnas: Optional[List[str]] = None,
        tipo_diagonal: str = "kde",
    ) -> sns.PairGrid:
        """Pair plot de seaborn para análisis multivariado."""
        df_plot = df[columnas].copy() if columnas else df.select_dtypes(include=np.number).copy()
        if hue and hue not in df_plot.columns:
            df_plot[hue] = df[hue]
        g = sns.pairplot(df_plot, hue=hue, diag_kind=tipo_diagonal, plot_kws={"alpha": 0.5})
        g.figure.suptitle("Pair Plot", y=1.02, fontsize=13, fontweight="bold")
        return g

    def scatter_con_regresion(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        figsize: Tuple = (7, 5),
    ) -> plt.Figure:
        """Dispersión con línea de regresión lineal."""
        fig, ax = plt.subplots(figsize=figsize)
        sns.regplot(data=df, x=x, y=y, scatter_kws={"alpha": 0.4, "s": 20}, ax=ax)
        ax.set_title(f"{y} vs {x}")
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICADOR CATEGÓRICO
# ──────────────────────────────────────────────────────────────────────────────

class CategoricalPlotter:
    """Visualiza variables categóricas."""

    def grafico_barras(
        self,
        df: pd.DataFrame,
        x: str,
        y: Optional[str] = None,
        top_n: int = 15,
        figsize: Tuple = (10, 5),
        titulo: Optional[str] = None,
    ) -> plt.Figure:
        """Gráfico de barras horizontal de conteos o valores agregados."""
        if y is None:
            conteos = df[x].value_counts().head(top_n)
            datos = conteos.reset_index()
            datos.columns = [x, "conteo"]
            col_y = "conteo"
        else:
            datos = df.groupby(x)[y].mean().nlargest(top_n).reset_index()
            col_y = y

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=datos, y=x, x=col_y, ax=ax, palette=PALETA)
        ax.set_title(titulo or f"Top {top_n} {x}")
        ax.set_xlabel(col_y)
        ax.set_ylabel(x)
        fig.tight_layout()
        return fig

    def grafico_donut(
        self,
        serie: pd.Series,
        top_n: int = 8,
        figsize: Tuple = (7, 7),
        titulo: Optional[str] = None,
    ) -> plt.Figure:
        """Gráfico donut para proporciones categóricas."""
        conteos = serie.value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=figsize)
        props_cuña = {"width": 0.5, "edgecolor": "white", "linewidth": 2}
        colores = sns.color_palette(PALETA, len(conteos))
        ax.pie(
            conteos.values,
            labels=conteos.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colores,
            wedgeprops=props_cuña,
        )
        ax.set_title(titulo or f"Distribución de {serie.name or 'Categoría'}")
        fig.tight_layout()
        return fig

    def barras_apiladas(
        self,
        df: pd.DataFrame,
        x: str,
        hue: str,
        figsize: Tuple = (12, 5),
        normalizar: bool = True,
    ) -> plt.Figure:
        """Gráfico de barras apiladas mostrando composición de categorías."""
        ct = pd.crosstab(df[x], df[hue], normalize="index" if normalizar else False)
        fig, ax = plt.subplots(figsize=figsize)
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.75)
        ax.set_title(f"Composición de {hue} por {x}")
        ax.set_ylabel("Proporción" if normalizar else "Conteo")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=8)
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICADOR DE SERIES DE TIEMPO
# ──────────────────────────────────────────────────────────────────────────────

class TimeSeriesPlotter:
    """Visualiza datos de series de tiempo."""

    def grafico_linea(
        self,
        serie: pd.Series,
        ventana_movil: Optional[int] = None,
        titulo: Optional[str] = None,
        figsize: Tuple = (12, 4),
    ) -> plt.Figure:
        """Gráfico de línea con promedio móvil opcional superpuesto."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(serie.index, serie.values, alpha=0.5, color="#5B84C4", lw=1, label="Original")
        if ventana_movil:
            movil = serie.rolling(window=ventana_movil).mean()
            ax.plot(movil.index, movil.values, color="#E07B54", lw=2,
                    label=f"Media Móvil ({ventana_movil})")
            ax.legend()
        ax.set_title(titulo or f"Serie de Tiempo: {serie.name or ''}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor")
        fig.tight_layout()
        return fig

    def grafico_estacional(
        self,
        serie: pd.Series,
        frecuencia: str = "M",
        figsize: Tuple = (12, 5),
    ) -> plt.Figure:
        """Box plot de valores agrupados por período (mes, día de la semana, etc.)."""
        df_tmp = pd.DataFrame({"valor": serie})
        if frecuencia == "M":
            df_tmp["periodo"] = serie.index.month
            etiqueta_x = "Mes"
        elif frecuencia == "DOW":
            df_tmp["periodo"] = serie.index.dayofweek
            etiqueta_x = "Día de la Semana"
        elif frecuencia == "H":
            df_tmp["periodo"] = serie.index.hour
            etiqueta_x = "Hora"
        else:
            df_tmp["periodo"] = serie.index.year
            etiqueta_x = "Año"
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=df_tmp, x="periodo", y="valor", ax=ax, palette=PALETA)
        ax.set_title(f"Patrón Estacional por {etiqueta_x}")
        ax.set_xlabel(etiqueta_x)
        ax.set_ylabel(serie.name or "Valor")
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICADOR DE DESEMPEÑO DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────

class ModelPerformancePlotter:
    """Visualiza métricas de evaluación de modelos de ML."""

    def curva_roc(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        nombre_modelo: str = "Modelo",
        figsize: Tuple = (6, 5),
    ) -> plt.Figure:
        """Grafica la curva ROC."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, lw=2, color="#5B84C4", label=f"{nombre_modelo} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador Aleatorio")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#5B84C4")
        ax.set_xlabel("Tasa de Falsos Positivos")
        ax.set_ylabel("Tasa de Verdaderos Positivos")
        ax.set_title("Curva ROC")
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig

    def grafico_matriz_confusion(
        self,
        cm: pd.DataFrame,
        normalizar: bool = True,
        figsize: Tuple = (6, 5),
    ) -> plt.Figure:
        """Mapa de calor de la matriz de confusión."""
        datos = cm.div(cm.sum(axis=1), axis=0).round(2) if normalizar else cm
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            datos, annot=True, fmt=".2f" if normalizar else "d",
            cmap="Blues", linewidths=0.5, ax=ax,
        )
        ax.set_title("Matriz de Confusión" + (" (Normalizada)" if normalizar else ""))
        ax.set_ylabel("Real")
        ax.set_xlabel("Predicho")
        fig.tight_layout()
        return fig

    def grafico_residuos(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple = (12, 4),
    ) -> plt.Figure:
        """Análisis de residuos: residuos vs ajustados y distribución de residuos."""
        residuos = y_true - y_pred
        fig, ejes = plt.subplots(1, 2, figsize=figsize)
        ejes[0].scatter(y_pred, residuos, alpha=0.4, color="#5B84C4", s=15)
        ejes[0].axhline(0, color="red", ls="--", lw=1)
        ejes[0].set_xlabel("Valores Ajustados")
        ejes[0].set_ylabel("Residuos")
        ejes[0].set_title("Residuos vs Ajustados")
        sns.histplot(residuos, kde=True, ax=ejes[1], color="#E07B54")
        ejes[1].set_title("Distribución de Residuos")
        ejes[1].set_xlabel("Residuo")
        fig.tight_layout()
        return fig

    def importancia_caracteristicas(
        self,
        nombres: List[str],
        importancias: np.ndarray,
        top_n: int = 20,
        figsize: Tuple = (8, 6),
    ) -> plt.Figure:
        """Gráfico de barras horizontal de importancia de características."""
        df = (
            pd.DataFrame({"caracteristica": nombres, "importancia": importancias})
            .sort_values("importancia", ascending=True)
            .tail(top_n)
        )
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df["caracteristica"], df["importancia"], color="#5B84C4")
        ax.set_title(f"Top {top_n} Características más Importantes")
        ax.set_xlabel("Importancia")
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCTOR DE DASHBOARDS
# ──────────────────────────────────────────────────────────────────────────────

class DashboardBuilder:
    """Compone dashboards de diagnóstico multipanel."""

    def dashboard_eda(
        self, df: pd.DataFrame, numeric_cols: List[str], cat_col: Optional[str] = None
    ) -> plt.Figure:
        """
        Genera automáticamente un dashboard EDA 2×3:
        Fila 1: distribuciones de las primeras 3 columnas numéricas
        Fila 2: mapa de calor de correlación + barras de categoría + box plots combinados
        """
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # Fila 1: distribuciones
        for i, col in enumerate(numeric_cols[:3]):
            ax = fig.add_subplot(gs[0, i])
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribución: {col}", fontsize=9)
            ax.set_xlabel("")

        # Fila 2, col 0: mapa de calor de correlación (mini)
        ax_corr = fig.add_subplot(gs[1, 0])
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=len(numeric_cols) <= 6, fmt=".1f", cmap="coolwarm",
                    center=0, ax=ax_corr, cbar=False, square=True, linewidths=0.3)
        ax_corr.set_title("Matriz de Correlación", fontsize=9)

        # Fila 2, col 1: barras categóricas (si se proporcionó)
        ax_cat = fig.add_subplot(gs[1, 1])
        if cat_col and cat_col in df.columns:
            conteos = df[cat_col].value_counts().head(10)
            ax_cat.barh(conteos.index, conteos.values, color=sns.color_palette(PALETA, len(conteos)))
            ax_cat.set_title(f"Top Valores: {cat_col}", fontsize=9)
        else:
            ax_cat.axis("off")

        # Fila 2, col 2: box plots normalizados
        ax_box = fig.add_subplot(gs[1, 2])
        norm = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        norm.boxplot(ax=ax_box)
        ax_box.set_title("Box Plots Normalizados", fontsize=9)
        ax_box.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=7)

        fig.suptitle("Dashboard EDA", fontsize=14, fontweight="bold", y=1.01)
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Demostración
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    df = pd.DataFrame({
        "edad": np.random.normal(35, 10, 500),
        "ingresos": np.random.lognormal(10.5, 0.5, 500),
        "puntaje": np.random.beta(2, 5, 500) * 100,
        "categoria": np.random.choice(["A", "B", "C", "D", "E"], 500),
    })

    # Distribución
    dp = DistributionPlotter()
    fig1 = dp.histogram_kde(df["edad"], titulo="Distribución de Edad")

    # Correlación
    cp = CorrelationPlotter()
    fig2 = cp.heatmap(df, metodo="pearson")

    # Categórico
    catp = CategoricalPlotter()
    fig3 = catp.grafico_barras(df, x="categoria")

    # Dashboard
    builder = DashboardBuilder()
    fig4 = builder.dashboard_eda(df, numeric_cols=["edad", "ingresos", "puntaje"], cat_col="categoria")

    plt.show()
    print("Todos los gráficos generados exitosamente.")
