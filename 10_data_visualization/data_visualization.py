"""
Data Visualization Toolkit
============================
Publication-ready charts and dashboards using matplotlib and seaborn.
Follows SOLID principles with separate classes per visualization domain.

Modules:
  - DistributionPlotter      : histograms, KDE, Q-Q plots, violin plots
  - CorrelationPlotter       : heatmaps, pair plots, scatter matrices
  - CategoricalPlotter       : bar charts, count plots, pie/donut charts
  - TimeSeriesPlotter        : line plots, rolling averages, seasonality
  - ModelPerformancePlotter  : ROC, PR curves, confusion matrix, residuals
  - DashboardBuilder         : multi-panel figure composition

Author: Data Science Analytics Toolkit
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

# ── Default Style ──────────────────────────────────────────────────────────────
PALETTE = "husl"
STYLE = "whitegrid"
CONTEXT = "paper"
DPI = 120

sns.set_theme(style=STYLE, context=CONTEXT, palette=PALETTE)
plt.rcParams.update({"figure.dpi": DPI, "axes.titlepad": 12})


# ──────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION PLOTTER
# ──────────────────────────────────────────────────────────────────────────────

class DistributionPlotter:
    """Plot univariate and bivariate distributions."""

    def histogram_kde(
        self,
        series: pd.Series,
        bins: int = 30,
        title: Optional[str] = None,
        color: str = "#5B84C4",
        figsize: Tuple = (8, 4),
    ) -> plt.Figure:
        """Histogram with overlaid KDE curve."""
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(series.dropna(), bins=bins, kde=True, ax=ax, color=color)
        ax.set_title(title or f"Distribution of {series.name or 'Series'}")
        ax.set_xlabel(series.name or "Value")
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    def qqplot(
        self,
        series: pd.Series,
        title: Optional[str] = None,
        figsize: Tuple = (6, 5),
    ) -> plt.Figure:
        """Q-Q plot to assess normality."""
        fig, ax = plt.subplots(figsize=figsize)
        qq = stats.probplot(series.dropna(), dist="norm")
        theoretical_q = qq[0][0]
        sample_q = qq[0][1]
        ax.scatter(theoretical_q, sample_q, alpha=0.6, color="#E07B54", s=20)
        fit = np.polyfit(theoretical_q, sample_q, 1)
        ax.plot(theoretical_q, np.polyval(fit, theoretical_q), color="#333", lw=1.5, ls="--")
        ax.set_title(title or f"Q-Q Plot: {series.name or 'Series'}")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        fig.tight_layout()
        return fig

    def violin_box(
        self,
        df: pd.DataFrame,
        numeric_col: str,
        group_col: Optional[str] = None,
        figsize: Tuple = (10, 5),
    ) -> plt.Figure:
        """Violin + box plot optionally grouped by category."""
        fig, ax = plt.subplots(figsize=figsize)
        if group_col:
            sns.violinplot(data=df, x=group_col, y=numeric_col, ax=ax, inner="box", palette=PALETTE)
        else:
            sns.violinplot(data=df, y=numeric_col, ax=ax, inner="box", color="#5B84C4")
        ax.set_title(f"Distribution of {numeric_col}" + (f" by {group_col}" if group_col else ""))
        fig.tight_layout()
        return fig

    def multi_distribution(
        self, df: pd.DataFrame, columns: List[str], figsize: Tuple = (15, 4)
    ) -> plt.Figure:
        """Plot histograms for multiple numeric columns in a single row."""
        n = len(columns)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, columns):
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(col)
            ax.set_xlabel("")
        fig.suptitle("Variable Distributions", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# CORRELATION PLOTTER
# ──────────────────────────────────────────────────────────────────────────────

class CorrelationPlotter:
    """Visualize pairwise relationships and correlations."""

    def heatmap(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
        figsize: Tuple = (10, 8),
        annot: bool = True,
    ) -> plt.Figure:
        """
        Correlation heatmap.

        Parameters
        ----------
        df : pd.DataFrame
            Numeric DataFrame.
        method : str
            'pearson', 'spearman', or 'kendall'.
        """
        corr = df.select_dtypes(include=np.number).corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr, mask=mask, annot=annot, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax,
        )
        ax.set_title(f"{method.capitalize()} Correlation Matrix", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def pairplot(
        self,
        df: pd.DataFrame,
        hue: Optional[str] = None,
        columns: Optional[List[str]] = None,
        diag_kind: str = "kde",
    ) -> sns.PairGrid:
        """Seaborn pair plot for multivariate analysis."""
        plot_df = df[columns].copy() if columns else df.select_dtypes(include=np.number).copy()
        if hue and hue not in plot_df.columns:
            plot_df[hue] = df[hue]
        g = sns.pairplot(plot_df, hue=hue, diag_kind=diag_kind, plot_kws={"alpha": 0.5})
        g.figure.suptitle("Pair Plot", y=1.02, fontsize=13, fontweight="bold")
        return g

    def scatter_with_regression(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        figsize: Tuple = (7, 5),
    ) -> plt.Figure:
        """Scatter plot with linear regression line."""
        fig, ax = plt.subplots(figsize=figsize)
        sns.regplot(data=df, x=x, y=y, scatter_kws={"alpha": 0.4, "s": 20}, ax=ax)
        ax.set_title(f"{y} vs {x}")
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORICAL PLOTTER
# ──────────────────────────────────────────────────────────────────────────────

class CategoricalPlotter:
    """Visualize categorical variables."""

    def bar_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: Optional[str] = None,
        top_n: int = 15,
        figsize: Tuple = (10, 5),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Horizontal bar chart of counts or aggregated values."""
        if y is None:
            counts = df[x].value_counts().head(top_n)
            data = counts.reset_index()
            data.columns = [x, "count"]
            y_col = "count"
        else:
            data = df.groupby(x)[y].mean().nlargest(top_n).reset_index()
            y_col = y

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=data, y=x, x=y_col, ax=ax, palette=PALETTE)
        ax.set_title(title or f"Top {top_n} {x}")
        ax.set_xlabel(y_col)
        ax.set_ylabel(x)
        fig.tight_layout()
        return fig

    def donut_chart(
        self,
        series: pd.Series,
        top_n: int = 8,
        figsize: Tuple = (7, 7),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Donut chart for categorical proportions."""
        counts = series.value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=figsize)
        wedge_props = {"width": 0.5, "edgecolor": "white", "linewidth": 2}
        colors = sns.color_palette(PALETTE, len(counts))
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops=wedge_props,
        )
        ax.set_title(title or f"{series.name or 'Category'} Distribution")
        fig.tight_layout()
        return fig

    def stacked_bar(
        self,
        df: pd.DataFrame,
        x: str,
        hue: str,
        figsize: Tuple = (12, 5),
        normalize: bool = True,
    ) -> plt.Figure:
        """Stacked bar chart showing category composition."""
        ct = pd.crosstab(df[x], df[hue], normalize="index" if normalize else False)
        fig, ax = plt.subplots(figsize=figsize)
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.75)
        ax.set_title(f"{hue} Composition by {x}")
        ax.set_ylabel("Proportion" if normalize else "Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=8)
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# TIME SERIES PLOTTER
# ──────────────────────────────────────────────────────────────────────────────

class TimeSeriesPlotter:
    """Visualize time series data."""

    def line_plot(
        self,
        series: pd.Series,
        rolling_window: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Tuple = (12, 4),
    ) -> plt.Figure:
        """Line plot with optional rolling average overlay."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(series.index, series.values, alpha=0.5, color="#5B84C4", lw=1, label="Original")
        if rolling_window:
            rolling = series.rolling(window=rolling_window).mean()
            ax.plot(rolling.index, rolling.values, color="#E07B54", lw=2,
                    label=f"Rolling Mean ({rolling_window})")
            ax.legend()
        ax.set_title(title or f"Time Series: {series.name or ''}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        fig.tight_layout()
        return fig

    def seasonal_plot(
        self,
        series: pd.Series,
        freq: str = "M",
        figsize: Tuple = (12, 5),
    ) -> plt.Figure:
        """Box plot of values grouped by period (month, day of week, etc.)."""
        df_tmp = pd.DataFrame({"value": series})
        if freq == "M":
            df_tmp["period"] = series.index.month
            xlabel = "Month"
        elif freq == "DOW":
            df_tmp["period"] = series.index.dayofweek
            xlabel = "Day of Week"
        elif freq == "H":
            df_tmp["period"] = series.index.hour
            xlabel = "Hour"
        else:
            df_tmp["period"] = series.index.year
            xlabel = "Year"
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=df_tmp, x="period", y="value", ax=ax, palette=PALETTE)
        ax.set_title(f"Seasonal Pattern by {xlabel}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(series.name or "Value")
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE PLOTTER
# ──────────────────────────────────────────────────────────────────────────────

class ModelPerformancePlotter:
    """Visualize ML model evaluation metrics."""

    def roc_curve_plot(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        model_name: str = "Model",
        figsize: Tuple = (6, 5),
    ) -> plt.Figure:
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, lw=2, color="#5B84C4", label=f"{model_name} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#5B84C4")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig

    def confusion_matrix_plot(
        self,
        cm: pd.DataFrame,
        normalize: bool = True,
        figsize: Tuple = (6, 5),
    ) -> plt.Figure:
        """Heatmap of confusion matrix."""
        data = cm.div(cm.sum(axis=1), axis=0).round(2) if normalize else cm
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data, annot=True, fmt=".2f" if normalize else "d",
            cmap="Blues", linewidths=0.5, ax=ax,
        )
        ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        fig.tight_layout()
        return fig

    def residuals_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple = (12, 4),
    ) -> plt.Figure:
        """Residual analysis: residuals vs fitted and residual distribution."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].scatter(y_pred, residuals, alpha=0.4, color="#5B84C4", s=15)
        axes[0].axhline(0, color="red", ls="--", lw=1)
        axes[0].set_xlabel("Fitted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Fitted")
        sns.histplot(residuals, kde=True, ax=axes[1], color="#E07B54")
        axes[1].set_title("Residual Distribution")
        axes[1].set_xlabel("Residual")
        fig.tight_layout()
        return fig

    def feature_importance_plot(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        figsize: Tuple = (8, 6),
    ) -> plt.Figure:
        """Horizontal bar chart of feature importances."""
        df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=True)
            .tail(top_n)
        )
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df["feature"], df["importance"], color="#5B84C4")
        ax.set_title(f"Top {top_n} Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD BUILDER
# ──────────────────────────────────────────────────────────────────────────────

class DashboardBuilder:
    """Compose multi-panel diagnostic dashboards."""

    def eda_dashboard(
        self, df: pd.DataFrame, numeric_cols: List[str], cat_col: Optional[str] = None
    ) -> plt.Figure:
        """
        Auto-generate a 2×3 EDA dashboard:
        Row 1: distributions of first 3 numeric columns
        Row 2: correlation heatmap + top category bar + combined box plots
        """
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # Row 1: distributions
        for i, col in enumerate(numeric_cols[:3]):
            ax = fig.add_subplot(gs[0, i])
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution: {col}", fontsize=9)
            ax.set_xlabel("")

        # Row 2, col 0: correlation heatmap (mini)
        ax_corr = fig.add_subplot(gs[1, 0])
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=len(numeric_cols) <= 6, fmt=".1f", cmap="coolwarm",
                    center=0, ax=ax_corr, cbar=False, square=True, linewidths=0.3)
        ax_corr.set_title("Correlation Matrix", fontsize=9)

        # Row 2, col 1: categorical bar (if provided)
        ax_cat = fig.add_subplot(gs[1, 1])
        if cat_col and cat_col in df.columns:
            counts = df[cat_col].value_counts().head(10)
            ax_cat.barh(counts.index, counts.values, color=sns.color_palette(PALETTE, len(counts)))
            ax_cat.set_title(f"Top Values: {cat_col}", fontsize=9)
        else:
            ax_cat.axis("off")

        # Row 2, col 2: box plots
        ax_box = fig.add_subplot(gs[1, 2])
        normed = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        normed.boxplot(ax=ax_box)
        ax_box.set_title("Normalized Box Plots", fontsize=9)
        ax_box.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=7)

        fig.suptitle("EDA Dashboard", fontsize=14, fontweight="bold", y=1.01)
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.normal(35, 10, 500),
        "income": np.random.lognormal(10.5, 0.5, 500),
        "score": np.random.beta(2, 5, 500) * 100,
        "category": np.random.choice(["A", "B", "C", "D", "E"], 500),
    })

    # Distribution
    dp = DistributionPlotter()
    fig1 = dp.histogram_kde(df["age"], title="Age Distribution")

    # Correlation
    cp = CorrelationPlotter()
    fig2 = cp.heatmap(df, method="pearson")

    # Categorical
    catp = CategoricalPlotter()
    fig3 = catp.bar_chart(df, x="category")

    # Dashboard
    builder = DashboardBuilder()
    fig4 = builder.eda_dashboard(df, numeric_cols=["age", "income", "score"], cat_col="category")

    plt.show()
    print("All plots generated successfully.")
