"""
Analizador de Series de Tiempo
================================
Kit de herramientas completo para análisis de series de tiempo:
descomposición, pruebas de estacionariedad, modelos de pronóstico
y detección de anomalías.

Autor: Dody Dueñas
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_DISPONIBLE = True
except ImportError:
    STATSMODELS_DISPONIBLE = False


class PruebaEstacionariedad:
    """Pruebas de estacionariedad usando ADF, KPSS y estadísticas de ventana móvil."""

    def prueba_adf(self, serie: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Prueba Dickey-Fuller Aumentada (ADF) para raíz unitaria.

        Parámetros
        ----------
        serie : pd.Series
            Serie de tiempo a evaluar.
        alpha : float
            Nivel de significancia (por defecto 0.05).

        Retorna
        -------
        dict
            Estadístico, p-valor, valores críticos y veredicto de estacionariedad.
        """
        if not STATSMODELS_DISPONIBLE:
            raise ImportError("Se requiere statsmodels para la prueba ADF.")
        resultado = adfuller(serie.dropna(), autolag="AIC")
        salida = {
            "estadistico": resultado[0],
            "p_valor": resultado[1],
            "n_rezagos": resultado[2],
            "n_observaciones": resultado[3],
            "valores_criticos": resultado[4],
            "es_estacionaria": resultado[1] < alpha,
            "interpretacion": (
                "La serie es estacionaria (se rechaza H0)" if resultado[1] < alpha
                else "La serie NO es estacionaria (no se rechaza H0)"
            ),
        }
        return salida

    def prueba_kpss(self, serie: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Prueba KPSS de estacionariedad de nivel o tendencia.

        Parámetros
        ----------
        serie : pd.Series
            Serie de tiempo a evaluar.
        alpha : float
            Nivel de significancia.

        Retorna
        -------
        dict
            Estadístico, p-valor, valores críticos y veredicto.
        """
        if not STATSMODELS_DISPONIBLE:
            raise ImportError("Se requiere statsmodels para la prueba KPSS.")
        stat, p_valor, n_rezagos, valores_criticos = kpss(serie.dropna(), regression="c", nlags="auto")
        salida = {
            "estadistico": stat,
            "p_valor": p_valor,
            "n_rezagos": n_rezagos,
            "valores_criticos": valores_criticos,
            "es_estacionaria": p_valor > alpha,
            "interpretacion": (
                "La serie es estacionaria (no se rechaza H0)" if p_valor > alpha
                else "La serie NO es estacionaria (se rechaza H0)"
            ),
        }
        return salida

    def estadisticas_moviles(
        self, serie: pd.Series, ventana: int = 12
    ) -> pd.DataFrame:
        """
        Calcula media y desviación estándar móviles para evaluación visual de estacionariedad.

        Parámetros
        ----------
        serie : pd.Series
            Serie de tiempo.
        ventana : int
            Tamaño de la ventana móvil.

        Retorna
        -------
        pd.DataFrame
            DataFrame con la serie original, media móvil y desv. estándar móvil.
        """
        df = pd.DataFrame({"original": serie})
        df["media_movil"] = serie.rolling(window=ventana).mean()
        df["std_movil"] = serie.rolling(window=ventana).std()
        return df

    def informe_completo(self, serie: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """Ejecuta ADF y KPSS y resume la estacionariedad."""
        adf = self.prueba_adf(serie, alpha)
        kpss_res = self.prueba_kpss(serie, alpha)
        ambas_estacionarias = adf["es_estacionaria"] and kpss_res["es_estacionaria"]
        return {
            "adf": adf,
            "kpss": kpss_res,
            "ambas_confirman_estacionariedad": ambas_estacionarias,
            "summary": (
                "Ambas pruebas confirman estacionariedad." if ambas_estacionarias
                else "Las pruebas no coinciden o la serie no es estacionaria — considere aplicar diferenciación."
            ),
        }


class DescompositorSeries:
    """Descompone la serie de tiempo en componentes de tendencia, estacionalidad y residuo."""

    def descomposicion_clasica(
        self,
        serie: pd.Series,
        periodo: int,
        modelo: str = "additive",
    ) -> Any:
        """
        Descomposición estacional clásica (media móvil).

        Parámetros
        ----------
        serie : pd.Series
            Serie de tiempo con índice DatetimeIndex.
        periodo : int
            Período de la estacionalidad (ej: 12 para mensual, 7 para semanal).
        modelo : str
            'additive' o 'multiplicative'.

        Retorna
        -------
        DecomposeResult
            Resultado de descomposición con tendencia, estacionalidad y residuo.
        """
        if not STATSMODELS_DISPONIBLE:
            raise ImportError("Se requiere statsmodels.")
        return seasonal_decompose(serie, period=periodo, model=modelo, extrapolate_trend="freq")

    def descomposicion_stl(
        self,
        serie: pd.Series,
        periodo: int,
        robusto: bool = True,
    ) -> Any:
        """
        Descomposición STL (Seasonal-Trend using LOESS) — más robusta que la clásica.

        Parámetros
        ----------
        serie : pd.Series
            Serie de tiempo.
        periodo : int
            Período de estacionalidad.
        robusto : bool
            Usar ajuste robusto para manejar valores atípicos.
        """
        if not STATSMODELS_DISPONIBLE:
            raise ImportError("Se requiere statsmodels.")
        stl = STL(serie, period=periodo, robust=robusto)
        return stl.fit()

    def extraer_componentes(self, descomposicion: Any) -> pd.DataFrame:
        """Extrae tendencia, estacionalidad y residuo en un DataFrame ordenado."""
        return pd.DataFrame(
            {
                "tendencia": descomposicion.trend,
                "estacionalidad": descomposicion.seasonal,
                "residuo": descomposicion.resid,
            }
        )


class PronosticadorARIMA:
    """Ajusta y pronostica usando ARIMA."""

    def __init__(self):
        self.modelo = None
        self.ajustado = None
        self.orden = None

    def ajustar(
        self,
        serie: pd.Series,
        orden: Tuple[int, int, int] = (1, 1, 1),
    ) -> "PronosticadorARIMA":
        """
        Ajusta un modelo ARIMA a la serie de tiempo.

        Parámetros
        ----------
        serie : pd.Series
            Serie de tiempo univariada.
        orden : tuple
            Orden (p, d, q) para ARIMA.

        Retorna
        -------
        self
        """
        if not STATSMODELS_DISPONIBLE:
            raise ImportError("Se requiere statsmodels.")
        self.orden = orden
        self.modelo = ARIMA(serie, order=orden)
        self.ajustado = self.modelo.fit()
        return self

    def pronosticar(self, pasos: int = 12) -> pd.Series:
        """
        Genera pronósticos futuros.

        Parámetros
        ----------
        pasos : int
            Número de períodos a pronosticar.

        Retorna
        -------
        pd.Series
            Valores pronosticados.
        """
        if self.ajustado is None:
            raise ValueError("Modelo no ajustado. Llame a ajustar() primero.")
        return self.ajustado.forecast(steps=pasos)

    def resumen(self) -> str:
        """Retorna el resumen del modelo como texto."""
        if self.ajustado is None:
            raise ValueError("Modelo no ajustado. Llame a ajustar() primero.")
        return str(self.ajustado.summary())

    def diagnostico_residuos(self) -> Dict[str, float]:
        """Diagnóstico básico sobre los residuos del modelo."""
        if self.ajustado is None:
            raise ValueError("Modelo no ajustado.")
        resid = self.ajustado.resid
        return {
            "media_residuo": float(resid.mean()),
            "std_residuo": float(resid.std()),
            "p_ljung_box": float(self.ajustado.test_serial_correlation("ljungbox", lags=10)[0, 1]),
        }


class PronosticadorSuavizadoExponencial:
    """Suavizado exponencial de Holt-Winters para tendencia + estacionalidad."""

    def __init__(self):
        self.ajustado = None

    def ajustar(
        self,
        serie: pd.Series,
        tendencia: Optional[str] = "add",
        estacionalidad: Optional[str] = "add",
        periodos_estacionales: int = 12,
        tendencia_amortiguada: bool = False,
    ) -> "PronosticadorSuavizadoExponencial":
        """
        Ajusta el modelo Holt-Winters.

        Parámetros
        ----------
        serie : pd.Series
            Datos de la serie de tiempo.
        tendencia : str o None
            'add', 'mul' o None.
        estacionalidad : str o None
            'add', 'mul' o None.
        periodos_estacionales : int
            Longitud del ciclo estacional.
        tendencia_amortiguada : bool
            Si se debe amortigurar la tendencia.

        Retorna
        -------
        self
        """
        if not STATSMODELS_DISPONIBLE:
            raise ImportError("Se requiere statsmodels.")
        modelo = ExponentialSmoothing(
            serie,
            trend=tendencia,
            seasonal=estacionalidad,
            seasonal_periods=periodos_estacionales,
            damped_trend=tendencia_amortiguada,
        )
        self.ajustado = modelo.fit(optimized=True)
        return self

    def pronosticar(self, pasos: int = 12) -> pd.Series:
        """Genera pronósticos futuros."""
        if self.ajustado is None:
            raise ValueError("Modelo no ajustado. Llame a ajustar() primero.")
        return self.ajustado.forecast(pasos)

    def parametros_suavizado(self) -> Dict[str, float]:
        """Retorna los parámetros de suavizado ajustados."""
        return {
            "alpha": self.ajustado.params.get("smoothing_level"),
            "beta": self.ajustado.params.get("smoothing_trend"),
            "gamma": self.ajustado.params.get("smoothing_seasonal"),
            "phi": self.ajustado.params.get("damping_trend"),
        }


class DetectorAnomaliasSeries:
    """Detecta anomalías en series de tiempo usando métodos estadísticos."""

    def anomalias_zscore(
        self, serie: pd.Series, umbral: float = 3.0
    ) -> pd.Series:
        """
        Marca anomalías donde |z-score| > umbral.

        Parámetros
        ----------
        serie : pd.Series
            Serie de tiempo.
        umbral : float
            Valor de corte del z-score.

        Retorna
        -------
        pd.Series
            Máscara booleana de posiciones anómalas.
        """
        z_scores = (serie - serie.mean()) / serie.std()
        return z_scores.abs() > umbral

    def anomalias_iqr(self, serie: pd.Series, factor: float = 1.5) -> pd.Series:
        """
        Marca anomalías fuera de Q1 - factor*IQR o Q3 + factor*IQR.

        Parámetros
        ----------
        serie : pd.Series
            Serie de tiempo.
        factor : float
            Multiplicador del IQR (1.5 = estándar, 3.0 = solo extremos).

        Retorna
        -------
        pd.Series
            Máscara booleana de posiciones anómalas.
        """
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        inferior = q1 - factor * iqr
        superior = q3 + factor * iqr
        return (serie < inferior) | (serie > superior)

    def anomalias_moviles(
        self, serie: pd.Series, ventana: int = 30, sigma: float = 2.5
    ) -> pd.Series:
        """
        Detecta anomalías respecto a media móvil ± sigma * desv. estándar móvil.

        Parámetros
        ----------
        serie : pd.Series
            Serie de tiempo.
        ventana : int
            Tamaño de la ventana móvil.
        sigma : float
            Número de desviaciones estándar para la banda.

        Retorna
        -------
        pd.Series
            Máscara booleana donde True = anomalía.
        """
        media_movil = serie.rolling(window=ventana, center=True).mean()
        std_movil = serie.rolling(window=ventana, center=True).std()
        banda_superior = media_movil + sigma * std_movil
        banda_inferior = media_movil - sigma * std_movil
        return (serie > banda_superior) | (serie < banda_inferior)

    def resumen_anomalias(
        self, serie: pd.Series, metodo: str = "zscore"
    ) -> pd.DataFrame:
        """
        Retorna un DataFrame con las marcas de tiempo y valores anómalos.

        Parámetros
        ----------
        serie : pd.Series
            Serie de tiempo.
        metodo : str
            'zscore', 'iqr' o 'movil'.

        Retorna
        -------
        pd.DataFrame
        """
        if metodo == "zscore":
            mascara = self.anomalias_zscore(serie)
        elif metodo == "iqr":
            mascara = self.anomalias_iqr(serie)
        elif metodo == "movil":
            mascara = self.anomalias_moviles(serie)
        else:
            raise ValueError(f"Método desconocido '{metodo}'. Use 'zscore', 'iqr' o 'movil'.")
        anomalias = serie[mascara]
        return pd.DataFrame({"timestamp": anomalias.index, "valor": anomalias.values})


class PipelineSeries:
    """
    Pipeline completo de series de tiempo:
      1. Prueba de estacionariedad
      2. Descomposición
      3. Pronóstico ARIMA
      4. Detección de anomalías
    """

    def __init__(self, serie: pd.Series, periodo: int = 12):
        self.serie = serie
        self.periodo = periodo
        self.prueba = PruebaEstacionariedad()
        self.descompositor = DescompositorSeries()
        self.pronosticador = PronosticadorARIMA()
        self.detector = DetectorAnomaliasSeries()

    def ejecutar(self, pasos_pronostico: int = 12, orden_arima: Tuple = (1, 1, 1)) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo.

        Retorna
        -------
        dict
            Resultados de cada etapa.
        """
        resultados = {}

        # 1. Estacionariedad
        resultados["estacionariedad"] = self.prueba.informe_completo(self.serie)

        # 2. Descomposición
        descomp = self.descompositor.descomposicion_clasica(self.serie, periodo=self.periodo)
        resultados["descomposicion"] = self.descompositor.extraer_componentes(descomp)

        # 3. Pronóstico ARIMA
        self.pronosticador.ajustar(self.serie, orden=orden_arima)
        resultados["pronostico"] = self.pronosticador.pronosticar(pasos=pasos_pronostico)

        # 4. Detección de anomalías
        resultados["anomalias"] = self.detector.resumen_anomalias(self.serie)

        return resultados


# ──────────────────────────────────────────────────────────────────────────────
# Demostración rápida
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generar serie de tiempo mensual sintética
    np.random.seed(42)
    fechas = pd.date_range("2018-01-01", periods=120, freq="MS")
    tendencia = np.linspace(100, 200, 120)
    estacionalidad = 20 * np.sin(2 * np.pi * np.arange(120) / 12)
    ruido = np.random.normal(0, 5, 120)
    serie = pd.Series(tendencia + estacionalidad + ruido, index=fechas, name="ventas")

    pipeline = PipelineSeries(serie, periodo=12)
    resultados = pipeline.ejecutar(pasos_pronostico=12, orden_arima=(1, 1, 1))

    print("=== Estacionariedad ===")
    print(resultados["estacionariedad"]["summary"])

    print("\n=== Pronóstico ARIMA (próximos 12 meses) ===")
    print(resultados["pronostico"].round(2).to_string())

    print("\n=== Anomalías Detectadas ===")
    print(resultados["anomalias"])
