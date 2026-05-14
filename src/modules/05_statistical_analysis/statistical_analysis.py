"""
📐 Statistical Analysis Toolkit
=================================
Herramientas para análisis estadístico descriptivo e inferencial.

Incluye:
- Estadística descriptiva extendida
- Pruebas de normalidad (Shapiro-Wilk, Kolmogorov-Smirnov)
- Pruebas de hipótesis (t-test, ANOVA, Chi-cuadrado, Mann-Whitney)
- Análisis de correlación con significancia estadística
- Intervalos de confianza
- Análisis de regresión lineal con diagnóstico
- Visualizaciones estadísticas profesionales

Autor: Dody Dueñas
Fecha: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import (
    shapiro, kstest, normaltest,
    ttest_1samp, ttest_ind, ttest_rel,
    mannwhitneyu, wilcoxon, kruskal,
    f_oneway, chi2_contingency,
    pearsonr, spearmanr, kendalltau
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_theme(style="whitegrid", palette="muted")


# ══════════════════════════════════════════════════════════════
#  1. ESTADÍSTICA DESCRIPTIVA EXTENDIDA
# ══════════════════════════════════════════════════════════════

def extended_describe(data: pd.Series, name: str = None) -> pd.Series:
    """
    Estadísticas descriptivas extendidas para una Serie numérica.

    Incluye: media, mediana, moda, varianza, desviación estándar,
    coeficiente de variación, asimetría (skewness), curtosis (kurtosis),
    percentiles Q1, Q3, IQR, y rango intercuartílico ajustado.
    """
    label = name or (data.name or 'variable')
    d = data.dropna()
    mode_val = d.mode()

    stats_dict = {
        'n': len(d),
        'n_missing': data.isnull().sum(),
        'mean': d.mean(),
        'median': d.median(),
        'mode': mode_val.iloc[0] if not mode_val.empty else np.nan,
        'std': d.std(),
        'variance': d.var(),
        'cv_%': d.std() / d.mean() * 100 if d.mean() != 0 else np.nan,
        'min': d.min(),
        'Q1 (25%)': d.quantile(0.25),
        'Q3 (75%)': d.quantile(0.75),
        'max': d.max(),
        'IQR': d.quantile(0.75) - d.quantile(0.25),
        'range': d.max() - d.min(),
        'skewness': d.skew(),
        'kurtosis': d.kurtosis(),
    }

    result = pd.Series(stats_dict, name=label).round(4)
    print(f"\n📐 Estadísticas Descriptivas: {label}")
    print("─" * 40)
    for k, v in stats_dict.items():
        print(f"  {k:<20} {v:>12.4f}" if isinstance(v, float) else f"  {k:<20} {v:>12}")
    return result


# ══════════════════════════════════════════════════════════════
#  2. PRUEBAS DE NORMALIDAD
# ══════════════════════════════════════════════════════════════

def test_normality(data: pd.Series, alpha: float = 0.05) -> dict:
    """
    Aplica múltiples pruebas de normalidad.

    Pruebas:
    - Shapiro-Wilk (n < 5000)
    - Kolmogorov-Smirnov
    - D'Agostino-Pearson

    Parámetros
    ----------
    alpha : float
        Nivel de significancia (default: 0.05)

    Returns
    -------
    dict con resultados de cada prueba.
    """
    d = data.dropna()
    results = {}

    print(f"\n🔬 Pruebas de Normalidad (α = {alpha})")
    print("─" * 55)

    # Shapiro-Wilk (recomendado n < 5000)
    if len(d) < 5000:
        stat, p = shapiro(d)
        results['shapiro_wilk'] = {'statistic': stat, 'p_value': p, 'normal': p > alpha}
        verdict = "✅ Normal" if p > alpha else "❌ No Normal"
        print(f"  Shapiro-Wilk:       W={stat:.4f}, p={p:.4f} → {verdict}")

    # Kolmogorov-Smirnov
    stat, p = kstest(d, 'norm', args=(d.mean(), d.std()))
    results['kolmogorov_smirnov'] = {'statistic': stat, 'p_value': p, 'normal': p > alpha}
    verdict = "✅ Normal" if p > alpha else "❌ No Normal"
    print(f"  Kolmogorov-Smirnov: D={stat:.4f}, p={p:.4f} → {verdict}")

    # D'Agostino-Pearson
    stat, p = normaltest(d)
    results['dagostino_pearson'] = {'statistic': stat, 'p_value': p, 'normal': p > alpha}
    verdict = "✅ Normal" if p > alpha else "❌ No Normal"
    print(f"  D'Agostino-Pearson: K²={stat:.4f}, p={p:.4f} → {verdict}")

    # Consenso
    normal_votes = sum(1 for r in results.values() if r['normal'])
    total_tests = len(results)
    consensus = "✅ DISTRIBUCIÓN NORMAL" if normal_votes > total_tests / 2 else "❌ NO NORMAL"
    print(f"\n  Consenso ({normal_votes}/{total_tests}): {consensus}")

    return results


# ══════════════════════════════════════════════════════════════
#  3. PRUEBAS DE HIPÓTESIS
# ══════════════════════════════════════════════════════════════

def one_sample_ttest(data: pd.Series, mu0: float, alpha: float = 0.05) -> dict:
    """
    T-Test de una muestra: H0: μ = mu0

    Parámetros
    ----------
    mu0 : float
        Valor de referencia de la hipótesis nula.
    """
    stat, p = ttest_1samp(data.dropna(), mu0)
    result = {
        'test': 'One-Sample T-Test',
        'H0': f'μ = {mu0}',
        't_statistic': stat,
        'p_value': p,
        'reject_H0': p < alpha,
        'conclusion': f"{'Rechazamos' if p < alpha else 'No rechazamos'} H0 con α={alpha}"
    }
    _print_hypothesis_result(result, alpha)
    return result


def two_sample_ttest(group1: pd.Series, group2: pd.Series, alpha: float = 0.05, equal_var: bool = False) -> dict:
    """
    T-Test de dos muestras independientes: H0: μ1 = μ2

    Parámetros
    ----------
    equal_var : bool
        Si True, asume varianzas iguales (Student). Si False, usa Welch.
    """
    stat, p = ttest_ind(group1.dropna(), group2.dropna(), equal_var=equal_var)
    test_name = "T-Test (Student)" if equal_var else "T-Test (Welch)"
    result = {
        'test': test_name,
        'H0': 'μ1 = μ2',
        't_statistic': stat,
        'p_value': p,
        'reject_H0': p < alpha,
        'mean_group1': group1.mean(),
        'mean_group2': group2.mean(),
        'conclusion': f"{'Rechazamos' if p < alpha else 'No rechazamos'} H0 con α={alpha}"
    }
    _print_hypothesis_result(result, alpha)
    return result


def anova_test(*groups, alpha: float = 0.05) -> dict:
    """
    ANOVA de una vía: H0: μ1 = μ2 = ... = μk (todas las medias son iguales)

    Parámetros
    ----------
    *groups : pd.Series
        Grupos a comparar. Mínimo 3.
    """
    clean_groups = [g.dropna() for g in groups]
    stat, p = f_oneway(*clean_groups)
    result = {
        'test': 'One-Way ANOVA',
        'H0': 'Todas las medias son iguales',
        'F_statistic': stat,
        'p_value': p,
        'reject_H0': p < alpha,
        'conclusion': f"{'Rechazamos' if p < alpha else 'No rechazamos'} H0 con α={alpha}"
    }
    _print_hypothesis_result(result, alpha)
    return result


def chi2_test(contingency_table: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Prueba Chi-cuadrado de independencia.

    Parámetros
    ----------
    contingency_table : pd.DataFrame
        Tabla de contingencia (frecuencias observadas).
    """
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    result = {
        'test': 'Chi-Cuadrado de Independencia',
        'H0': 'Las variables son independientes',
        'chi2_statistic': chi2,
        'p_value': p,
        'degrees_of_freedom': dof,
        'reject_H0': p < alpha,
        'conclusion': f"{'Rechazamos' if p < alpha else 'No rechazamos'} H0 con α={alpha}"
    }
    _print_hypothesis_result(result, alpha)
    return result


def mann_whitney_test(group1: pd.Series, group2: pd.Series, alpha: float = 0.05) -> dict:
    """
    Prueba Mann-Whitney U (alternativa no paramétrica al T-Test independiente).
    H0: Las distribuciones de ambos grupos son iguales.
    """
    stat, p = mannwhitneyu(group1.dropna(), group2.dropna(), alternative='two-sided')
    result = {
        'test': 'Mann-Whitney U',
        'H0': 'Las distribuciones son iguales',
        'U_statistic': stat,
        'p_value': p,
        'reject_H0': p < alpha,
        'conclusion': f"{'Rechazamos' if p < alpha else 'No rechazamos'} H0 con α={alpha}"
    }
    _print_hypothesis_result(result, alpha)
    return result


def _print_hypothesis_result(result: dict, alpha: float):
    """Helper para imprimir resultados de prueba de hipótesis."""
    print(f"\n🔬 {result['test']}")
    print(f"   H0: {result['H0']}")
    print(f"   Estadístico: {list(result.values())[3]:.4f}")
    print(f"   p-value: {result['p_value']:.6f}")
    print(f"   α: {alpha}")
    status = "🚨 RECHAZAR H0" if result['reject_H0'] else "✅ NO RECHAZAR H0"
    print(f"   → {status} | {result['conclusion']}")


# ══════════════════════════════════════════════════════════════
#  4. CORRELACIÓN CON SIGNIFICANCIA
# ══════════════════════════════════════════════════════════════

def correlation_analysis(df: pd.DataFrame, method: str = 'pearson', alpha: float = 0.05) -> pd.DataFrame:
    """
    Calcula la matriz de correlación con p-values y significancia.

    Parámetros
    ----------
    method : str
        'pearson', 'spearman', o 'kendall'
    alpha : float
        Nivel de significancia para marcar correlaciones significativas.

    Returns
    -------
    pd.DataFrame con correlaciones, p-values y significancia.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    cols = numeric_df.columns
    results = []

    corr_func = {'pearson': pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}[method]

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i >= j:
                continue
            r, p = corr_func(numeric_df[col1].dropna(), numeric_df[col2].dropna())
            results.append({
                'Variable 1': col1,
                'Variable 2': col2,
                'Correlación': round(r, 4),
                'p-value': round(p, 6),
                'Significativa': '✅' if p < alpha else '❌',
                'Fuerza': (
                    'Muy fuerte' if abs(r) > 0.8
                    else 'Fuerte' if abs(r) > 0.6
                    else 'Moderada' if abs(r) > 0.4
                    else 'Débil' if abs(r) > 0.2
                    else 'Muy débil'
                )
            })

    result_df = pd.DataFrame(results).sort_values('Correlación', key=abs, ascending=False)
    print(f"\n🔗 Análisis de Correlación ({method.capitalize()}, α={alpha}):")
    print(result_df.to_string(index=False))
    return result_df


# ══════════════════════════════════════════════════════════════
#  5. INTERVALOS DE CONFIANZA
# ══════════════════════════════════════════════════════════════

def confidence_interval(data: pd.Series, confidence: float = 0.95) -> tuple:
    """
    Calcula el intervalo de confianza para la media.

    Parámetros
    ----------
    confidence : float
        Nivel de confianza (default: 0.95 → IC del 95%)

    Returns
    -------
    (lower, upper, margin_of_error)
    """
    d = data.dropna()
    n = len(d)
    mean = d.mean()
    se = stats.sem(d)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    lower, upper = mean - h, mean + h

    print(f"\n📏 Intervalo de Confianza ({confidence*100:.0f}%)")
    print(f"   n = {n}")
    print(f"   Media = {mean:.4f}")
    print(f"   IC = [{lower:.4f}, {upper:.4f}]")
    print(f"   Margen de error = ±{h:.4f}")
    return lower, upper, h


# ══════════════════════════════════════════════════════════════
#  6. VISUALIZACIONES ESTADÍSTICAS
# ══════════════════════════════════════════════════════════════

def plot_normality_check(data: pd.Series, name: str = None):
    """
    Genera 4 gráficos para verificar normalidad:
    1. Histograma con curva normal
    2. Boxplot
    3. Q-Q Plot
    4. Violin Plot
    """
    label = name or (data.name or 'variable')
    d = data.dropna()

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Análisis de Normalidad: {label}', fontsize=16, y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # 1. Histograma con KDE y curva normal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(d, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    ax1.plot(x, stats.norm.pdf(x, d.mean(), d.std()), 'r-', lw=2, label='Normal teórica')
    d.plot(kind='kde', ax=ax1, color='darkblue', lw=2, label='KDE empírica')
    ax1.set_title('Histograma + KDE vs Normal Teórica', fontsize=12)
    ax1.legend()

    # 2. Boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot(d, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='navy'),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_title('Boxplot', fontsize=12)
    ax2.set_ylabel(label)

    # 3. Q-Q Plot
    ax3 = fig.add_subplot(gs[1, 0])
    (osm, osr), (slope, intercept, r) = stats.probplot(d, dist='norm')
    ax3.scatter(osm, osr, color='steelblue', alpha=0.7, s=20)
    ax3.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=2)
    ax3.set_title(f'Q-Q Plot (R² = {r**2:.4f})', fontsize=12)
    ax3.set_xlabel('Cuantiles teóricos')
    ax3.set_ylabel('Cuantiles observados')

    # 4. Violin Plot
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.violinplot(d, showmeans=True, showmedians=True)
    ax4.set_title('Violin Plot', fontsize=12)
    ax4.set_ylabel(label)

    plt.tight_layout()
    fname = f'normality_check_{label.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Guardado: {fname}")


def plot_hypothesis_comparison(group1: pd.Series, group2: pd.Series,
                                name1: str = 'Grupo 1', name2: str = 'Grupo 2'):
    """
    Visualiza la comparación de dos grupos con boxplot, violín y distribuciones.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Boxplot
    axes[0].boxplot([group1.dropna(), group2.dropna()],
                    labels=[name1, name2], patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))
    axes[0].set_title('Boxplot Comparativo', fontsize=13)

    # 2. Violin Plot
    axes[1].violinplot([group1.dropna(), group2.dropna()], showmeans=True, showmedians=True)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels([name1, name2])
    axes[1].set_title('Violin Plot Comparativo', fontsize=13)

    # 3. KDE Overlay
    group1.dropna().plot(kind='kde', ax=axes[2], label=name1, color='steelblue', lw=2)
    group2.dropna().plot(kind='kde', ax=axes[2], label=name2, color='crimson', lw=2)
    axes[2].axvline(group1.mean(), color='steelblue', linestyle='--', alpha=0.7)
    axes[2].axvline(group2.mean(), color='crimson', linestyle='--', alpha=0.7)
    axes[2].legend()
    axes[2].set_title('Distribuciones (KDE)', fontsize=13)

    fig.suptitle(f'Comparación: {name1} vs {name2}', fontsize=15)
    plt.tight_layout()
    plt.savefig('hypothesis_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────
#  Ejemplo de uso
# ──────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # Datos simulados
    ventas_A = pd.Series(np.random.normal(loc=5000, scale=800, size=200), name='Ventas_Canal_A')
    ventas_B = pd.Series(np.random.normal(loc=5300, scale=900, size=200), name='Ventas_Canal_B')

    print("=" * 60)
    print("📐  ANÁLISIS ESTADÍSTICO COMPLETO")
    print("=" * 60)

    # 1. Estadísticas descriptivas
    extended_describe(ventas_A)

    # 2. Normalidad
    test_normality(ventas_A)

    # 3. Intervalo de confianza
    confidence_interval(ventas_A)

    # 4. T-Test comparativo
    two_sample_ttest(ventas_A, ventas_B)

    # 5. Mann-Whitney (no paramétrico)
    mann_whitney_test(ventas_A, ventas_B)

    # 6. Visualizaciones
    plot_normality_check(ventas_A)
    plot_hypothesis_comparison(ventas_A, ventas_B, 'Canal A', 'Canal B')

    # 7. Correlación
    from sklearn.datasets import load_iris
    df_iris = load_iris(as_frame=True).frame
    correlation_analysis(df_iris)
