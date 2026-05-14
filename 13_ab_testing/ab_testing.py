"""
Kit de Pruebas A/B y Análisis de Experimentos
================================================
Framework estadístico para diseñar y analizar experimentos controlados:
- Cálculo del tamaño de muestra (análisis de potencia)
- Prueba Z y T para medias
- Prueba chi-cuadrado para proporciones/tasas de conversión
- Prueba Mann-Whitney U (no paramétrica)
- Pruebas A/B Bayesianas
- Análisis ANOVA multivariante (A/B/n)
- Generación de informes de experimentos

Autor: Dody Dueñas
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.special import betaln
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# CALCULADOR DE TAMAÑO DE MUESTRA
# ──────────────────────────────────────────────────────────────────────────────

class SampleSizeCalculator:
    """
    Calcula el tamaño de muestra requerido para el análisis de potencia de pruebas A/B.

    Basado en dos proporciones (tasas de conversión) o dos medias independientes.
    """

    def for_proportions(
        self,
        baseline_rate: float,
        min_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
        dos_colas: bool = True,
    ) -> Dict[str, Any]:
        """
        Calcula el tamaño de muestra para comparar dos proporciones.

        Parámetros
        ----------
        baseline_rate : float
            Tasa de conversión del grupo de control (ej: 0.10 para 10%).
        min_detectable_effect : float
            Incremento relativo mínimo a detectar (ej: 0.05 para +5%).
        alpha : float
            Nivel de significancia (Error Tipo I).
        power : float
            Potencia estadística (1 - Error Tipo II).
        dos_colas : bool
            Prueba de dos colas o una cola.

        Retorna
        -------
        dict con: n_por_grupo, total_n, tasa_tratamiento, tamanio_efecto
        """
        tasa_tratamiento = baseline_rate * (1 + min_detectable_effect)
        z_alpha = stats.norm.ppf(1 - alpha / (2 if dos_colas else 1))
        z_potencia = stats.norm.ppf(power)

        p_barra = (baseline_rate + tasa_tratamiento) / 2
        tamanio_efecto = abs(tasa_tratamiento - baseline_rate)

        n = (
            (z_alpha * np.sqrt(2 * p_barra * (1 - p_barra)) +
             z_potencia * np.sqrt(baseline_rate * (1 - baseline_rate) +
                                  tasa_tratamiento * (1 - tasa_tratamiento))) ** 2
            / tamanio_efecto ** 2
        )
        n = int(np.ceil(n))

        return {
            "n_por_grupo": n,
            "total_n": n * 2,
            "tasa_baseline": baseline_rate,
            "tasa_tratamiento": round(tasa_tratamiento, 6),
            "efecto_minimo_detectable": min_detectable_effect,
            "alpha": alpha,
            "potencia": power,
            "tamanio_efecto": round(tamanio_efecto, 6),
        }

    def for_means(
        self,
        media_baseline: float,
        std_baseline: float,
        delta_minimo: float,
        alpha: float = 0.05,
        power: float = 0.80,
        dos_colas: bool = True,
    ) -> Dict[str, Any]:
        """
        Calcula el tamaño de muestra para comparar dos medias (t-test independiente).

        Parámetros
        ----------
        media_baseline : float
            Media del grupo de control.
        std_baseline : float
            Estimación de la desviación estándar agrupada.
        delta_minimo : float
            Efecto mínimo absoluto a detectar.
        """
        z_alpha = stats.norm.ppf(1 - alpha / (2 if dos_colas else 1))
        z_potencia = stats.norm.ppf(power)
        n = int(np.ceil(2 * ((z_alpha + z_potencia) * std_baseline / delta_minimo) ** 2))
        return {
            "n_por_grupo": n,
            "total_n": n * 2,
            "media_baseline": media_baseline,
            "delta_minimo": delta_minimo,
            "d_cohen": round(delta_minimo / std_baseline, 4),
            "alpha": alpha,
            "potencia": power,
        }

    def efecto_minimo_detectable(
        self,
        n_por_grupo: int,
        tasa_baseline: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> float:
        """Dado un tamaño de muestra, calcula el EMD (Efecto Mínimo Detectable)."""
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_potencia = stats.norm.ppf(power)
        p = tasa_baseline
        emd = (z_alpha + z_potencia) * np.sqrt(2 * p * (1 - p) / n_por_grupo)
        return round(float(emd / p), 4)  # EMD relativo


# ──────────────────────────────────────────────────────────────────────────────
# TESTER A/B FRECUENTISTA
# ──────────────────────────────────────────────────────────────────────────────

class FrequentistABTester:
    """
    Pruebas de hipótesis frecuentistas para experimentos A/B.

    Soporta:
      - Prueba Z de dos proporciones (tasas de conversión)
      - T-test independiente de dos muestras (métricas continuas)
      - Prueba Mann-Whitney U (no paramétrica)
      - Prueba chi-cuadrado (tabla de contingencia multicelda)
    """

    def proportion_test(
        self,
        conversiones_control: int,
        n_control: int,
        conversiones_tratamiento: int,
        n_tratamiento: int,
        alpha: float = 0.05,
        dos_colas: bool = True,
    ) -> Dict[str, Any]:
        """
        Prueba Z de dos proporciones para comparar tasas de conversión.

        Parámetros
        ----------
        conversiones_control : int
            Número de éxitos en el grupo de control.
        n_control : int
            Total de usuarios en el grupo de control.
        conversiones_tratamiento : int
            Número de éxitos en el grupo de tratamiento.
        n_tratamiento : int
            Total de usuarios en el grupo de tratamiento.
        alpha : float
            Nivel de significancia.
        dos_colas : bool
            Prueba de dos colas (True) o una cola.

        Retorna
        -------
        dict con: tasa_control, tasa_tratamiento, incremento, estadistico_z,
                  p_valor, intervalo_confianza, es_significativo, recomendacion
        """
        p_c = conversiones_control / n_control
        p_t = conversiones_tratamiento / n_tratamiento
        p_pool = (conversiones_control + conversiones_tratamiento) / (n_control + n_tratamiento)

        ee = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_tratamiento))
        z_stat = (p_t - p_c) / ee
        p_valor = 2 * stats.norm.sf(abs(z_stat)) if dos_colas else stats.norm.sf(z_stat)

        # IC del 95% en la diferencia
        z_critico = stats.norm.ppf(1 - alpha / 2)
        ee_diff = np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_tratamiento)
        ic_bajo = (p_t - p_c) - z_critico * ee_diff
        ic_alto = (p_t - p_c) + z_critico * ee_diff

        significativo = p_valor < alpha
        incremento = (p_t - p_c) / p_c * 100

        return {
            "tasa_control": round(p_c, 6),
            "tasa_tratamiento": round(p_t, 6),
            "diferencia_absoluta": round(p_t - p_c, 6),
            "incremento_relativo_pct": round(incremento, 4),
            "estadistico_z": round(float(z_stat), 4),
            "p_valor": round(float(p_valor), 6),
            "intervalo_confianza_95pct": (round(ic_bajo, 6), round(ic_alto, 6)),
            "es_significativo": bool(significativo),
            "alpha": alpha,
            "recomendacion": (
                f"Implementar tratamiento — incremento estadísticamente significativo de {incremento:.2f}%."
                if significativo and incremento > 0
                else "No implementar — no hay mejora estadísticamente significativa."
                if not significativo
                else "El tratamiento es significativamente peor — no implementar."
            ),
        }

    def ttest(
        self,
        datos_control: Union[np.ndarray, pd.Series],
        datos_tratamiento: Union[np.ndarray, pd.Series],
        alpha: float = 0.05,
        varianzas_iguales: bool = False,
    ) -> Dict[str, Any]:
        """
        T-test de muestras independientes (Welch por defecto).

        Parámetros
        ----------
        datos_control : array-like
            Valores de la métrica para el grupo de control.
        datos_tratamiento : array-like
            Valores de la métrica para el grupo de tratamiento.
        alpha : float
            Nivel de significancia.
        varianzas_iguales : bool
            Usar t-test de Student (True) o Welch (False).
        """
        t_stat, p_valor = stats.ttest_ind(datos_control, datos_tratamiento, equal_var=varianzas_iguales)
        incremento = (np.mean(datos_tratamiento) - np.mean(datos_control)) / np.mean(datos_control) * 100

        # d de Cohen
        std_agrupada = np.sqrt((np.std(datos_control) ** 2 + np.std(datos_tratamiento) ** 2) / 2)
        d_cohen = (np.mean(datos_tratamiento) - np.mean(datos_control)) / std_agrupada

        # IC
        ic = stats.t.interval(
            1 - alpha,
            df=len(datos_control) + len(datos_tratamiento) - 2,
            loc=np.mean(datos_tratamiento) - np.mean(datos_control),
            scale=stats.sem(np.concatenate([datos_control, datos_tratamiento])),
        )

        return {
            "media_control": round(float(np.mean(datos_control)), 6),
            "media_tratamiento": round(float(np.mean(datos_tratamiento)), 6),
            "incremento_relativo_pct": round(float(incremento), 4),
            "estadistico_t": round(float(t_stat), 4),
            "p_valor": round(float(p_valor), 6),
            "d_cohen": round(float(d_cohen), 4),
            "tamanio_efecto": (
                "pequeño" if abs(d_cohen) < 0.5
                else "mediano" if abs(d_cohen) < 0.8
                else "grande"
            ),
            "intervalo_confianza_95pct": (round(ic[0], 6), round(ic[1], 6)),
            "es_significativo": bool(p_valor < alpha),
            "tipo_prueba": "t-test de Welch" if not varianzas_iguales else "t-test de Student",
        }

    def mann_whitney(
        self,
        datos_control: Union[np.ndarray, pd.Series],
        datos_tratamiento: Union[np.ndarray, pd.Series],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Prueba Mann-Whitney U no paramétrica."""
        u_stat, p_valor = stats.mannwhitneyu(
            datos_control, datos_tratamiento, alternative="two-sided"
        )
        return {
            "estadistico_u": round(float(u_stat), 4),
            "p_valor": round(float(p_valor), 6),
            "es_significativo": bool(p_valor < alpha),
            "mediana_control": round(float(np.median(datos_control)), 6),
            "mediana_tratamiento": round(float(np.median(datos_tratamiento)), 6),
            "tipo_prueba": "Mann-Whitney U (no paramétrica)",
        }

    def chi_square_test(
        self, tabla_contingencia: pd.DataFrame, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Prueba chi-cuadrado de independencia sobre una tabla de contingencia."""
        chi2, p_valor, gl, esperados = stats.chi2_contingency(tabla_contingencia)
        v_cramer = np.sqrt(chi2 / (tabla_contingencia.values.sum() * (min(tabla_contingencia.shape) - 1)))
        return {
            "estadistico_chi2": round(float(chi2), 4),
            "p_valor": round(float(p_valor), 6),
            "grados_libertad": int(gl),
            "v_cramer": round(float(v_cramer), 4),
            "es_significativo": bool(p_valor < alpha),
            "frecuencias_esperadas": pd.DataFrame(esperados, index=tabla_contingencia.index,
                                                   columns=tabla_contingencia.columns).round(2),
        }


# ──────────────────────────────────────────────────────────────────────────────
# TESTER A/B BAYESIANO
# ──────────────────────────────────────────────────────────────────────────────

class BayesianABTester:
    """
    Pruebas A/B Bayesianas para resultados binarios (conversiones).

    Utiliza el modelo conjugado Beta-Binomial.
    Prior: Beta(alpha_prior, beta_prior) — por defecto: Beta(1,1) = uniforme
    Posterior: Beta(alpha_prior + conversiones, beta_prior + no_conversiones)
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def analyze(
        self,
        conversiones_control: int,
        n_control: int,
        conversiones_tratamiento: int,
        n_tratamiento: int,
        n_muestras: int = 100_000,
    ) -> Dict[str, Any]:
        """
        Calcula distribuciones posteriores y P(tratamiento > control).

        Parámetros
        ----------
        conversiones_control, n_control : int
            Éxitos y total del grupo de control.
        conversiones_tratamiento, n_tratamiento : int
            Éxitos y total del grupo de tratamiento.
        n_muestras : int
            Muestras Monte Carlo para P(tratamiento > control).

        Retorna
        -------
        dict con estadísticas posteriores, P(tratamiento > control), incremento esperado
        """
        # Parámetros posteriores
        a_c = self.alpha_prior + conversiones_control
        b_c = self.beta_prior + (n_control - conversiones_control)
        a_t = self.alpha_prior + conversiones_tratamiento
        b_t = self.beta_prior + (n_tratamiento - conversiones_tratamiento)

        # Medias posteriores
        p_c = a_c / (a_c + b_c)
        p_t = a_t / (a_t + b_t)

        # Monte Carlo P(tratamiento > control)
        muestras_c = np.random.beta(a_c, b_c, n_muestras)
        muestras_t = np.random.beta(a_t, b_t, n_muestras)
        prob_t_gana = float((muestras_t > muestras_c).mean())

        # Distribución del incremento esperado
        incremento_esperado = muestras_t - muestras_c
        media_incremento = float(incremento_esperado.mean())
        ic_incremento = (float(np.percentile(incremento_esperado, 2.5)), float(np.percentile(incremento_esperado, 97.5)))

        return {
            "media_posterior_control": round(p_c, 6),
            "media_posterior_tratamiento": round(p_t, 6),
            "prob_tratamiento_gana": round(prob_t_gana, 4),
            "prob_control_gana": round(1 - prob_t_gana, 4),
            "media_incremento_esperado": round(media_incremento, 6),
            "ic_95pct_incremento": (round(ic_incremento[0], 6), round(ic_incremento[1], 6)),
            "recomendacion": (
                f"Desplegar tratamiento — {prob_t_gana:.1%} de probabilidad de mejora."
                if prob_t_gana >= 0.95
                else f"Recopilar más datos — {prob_t_gana:.1%} de probabilidad de que el tratamiento gane."
                if prob_t_gana >= 0.80
                else f"Se prefiere el control — solo {prob_t_gana:.1%} de probabilidad de que el tratamiento gane."
            ),
            "prior": f"Beta({self.alpha_prior}, {self.beta_prior})",
        }


# ──────────────────────────────────────────────────────────────────────────────
# GENERADOR DE INFORMES
# ──────────────────────────────────────────────────────────────────────────────

class ExperimentReporter:
    """Genera un informe completo de prueba A/B en Markdown."""

    def generate(
        self,
        nombre_experimento: str,
        resultado_frecuentista: Dict,
        resultado_bayesiano: Optional[Dict] = None,
        plan_tamanio_muestra: Optional[Dict] = None,
    ) -> str:
        """
        Produce un informe completo del experimento en formato Markdown.

        Retorna
        -------
        str — Informe formateado en Markdown
        """
        lineas = [
            f"# Informe de Prueba A/B: {nombre_experimento}",
            "",
            "## Resumen del Experimento",
            "",
        ]

        if plan_tamanio_muestra:
            lineas += [
                "### Análisis de Potencia Pre-Experimento",
                "",
                "| Parámetro | Valor |",
                "|-----------|-------|",
            ]
            for k, v in plan_tamanio_muestra.items():
                lineas.append(f"| {k} | {v} |")
            lineas.append("")

        lineas += [
            "## Resultados Frecuentistas",
            "",
            "| Métrica | Valor |",
            "|---------|-------|",
        ]
        for k, v in resultado_frecuentista.items():
            if k != "recomendacion":
                lineas.append(f"| {k} | {v} |")

        emoji_sig = "✅" if resultado_frecuentista.get("es_significativo") else "❌"
        lineas += [
            "",
            f"**Decisión ({emoji_sig}):** {resultado_frecuentista.get('recomendacion', '')}",
            "",
        ]

        if resultado_bayesiano:
            lineas += [
                "## Resultados Bayesianos",
                "",
                "| Métrica | Valor |",
                "|---------|-------|",
            ]
            for k, v in resultado_bayesiano.items():
                if k != "recomendacion":
                    lineas.append(f"| {k} | {v} |")
            lineas += [
                "",
                f"**Decisión Bayesiana:** {resultado_bayesiano.get('recomendacion', '')}",
                "",
            ]

        lineas += ["---", "*Informe generado por el Kit de Pruebas A/B — Autor: Dody Dueñas*"]
        return "\n".join(lineas)


# ──────────────────────────────────────────────────────────────────────────────
# Demostración
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # ── Planificación del tamaño de muestra ──────────────────────────────────
    calc = SampleSizeCalculator()
    plan = calc.for_proportions(baseline_rate=0.05, min_detectable_effect=0.20, power=0.80)
    print("=== Plan de Tamaño de Muestra ===")
    for k, v in plan.items():
        print(f"  {k}: {v}")

    # ── Prueba frecuentista de proporciones ───────────────────────────────────
    tester = FrequentistABTester()
    resultado_frec = tester.proportion_test(
        conversiones_control=500, n_control=10000,
        conversiones_tratamiento=580, n_tratamiento=10000,
    )
    print("\n=== Prueba Frecuentista de Proporciones ===")
    for k, v in resultado_frec.items():
        print(f"  {k}: {v}")

    # ── Análisis bayesiano ────────────────────────────────────────────────────
    bay_tester = BayesianABTester()
    resultado_bay = bay_tester.analyze(
        conversiones_control=500, n_control=10000,
        conversiones_tratamiento=580, n_tratamiento=10000,
    )
    print("\n=== Análisis Bayesiano ===")
    for k, v in resultado_bay.items():
        print(f"  {k}: {v}")

    # ── Informe ───────────────────────────────────────────────────────────────
    reporter = ExperimentReporter()
    informe_md = reporter.generate(
        nombre_experimento="Prueba de Color del Botón CTA en Página Principal",
        resultado_frecuentista=resultado_frec,
        resultado_bayesiano=resultado_bay,
        plan_tamanio_muestra=plan,
    )
    print("\n=== Vista Previa del Informe ===")
    print(informe_md[:800])
