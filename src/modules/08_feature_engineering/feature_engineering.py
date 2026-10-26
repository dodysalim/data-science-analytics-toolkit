"""
Kit de Ingeniería de Características
======================================
Librería SOLID para ingeniería de características automatizada y manual:
- Extracción de características de fecha/hora
- Características polinomiales e interacciones
- Codificación por objetivo y por frecuencia
- Discretización en intervalos (binning)
- Selección de características por filtros estadísticos
- Orquestación completa en pipeline

Autor: Dody Dueñas
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, chi2
)


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTOR DE CARACTERÍSTICAS FECHA/HORA
# ──────────────────────────────────────────────────────────────────────────────

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrae características temporales ricas a partir de columnas datetime.

    Características extraídas (configurables):
        año, mes, día, hora, minuto, día_semana, día_año,
        semana_año, trimestre, es_fin_semana, es_inicio_mes, es_fin_mes,
        codificaciones cíclicas sin/cos para mes, día_semana y hora.
    """

    def __init__(
        self,
        columnas: Optional[List[str]] = None,
        codificacion_ciclica: bool = True,
        eliminar_original: bool = False,
    ):
        self.columnas = columnas
        self.codificacion_ciclica = codificacion_ciclica
        self.eliminar_original = eliminar_original

    def fit(self, X: pd.DataFrame, y=None) -> "DateTimeFeatureExtractor":
        """Detecta columnas datetime si no se especifican."""
        if self.columnas is None:
            self._cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
        else:
            self._cols = self.columnas
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extrae características de todas las columnas datetime."""
        df = X.copy()
        for col in self._cols:
            dt = pd.to_datetime(df[col])
            df[f"{col}_año"] = dt.dt.year
            df[f"{col}_mes"] = dt.dt.month
            df[f"{col}_dia"] = dt.dt.day
            df[f"{col}_hora"] = dt.dt.hour
            df[f"{col}_dia_semana"] = dt.dt.dayofweek
            df[f"{col}_dia_año"] = dt.dt.dayofyear
            df[f"{col}_semana_año"] = dt.dt.isocalendar().week.astype(int)
            df[f"{col}_trimestre"] = dt.dt.quarter
            df[f"{col}_es_fin_semana"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
            df[f"{col}_es_inicio_mes"] = dt.dt.is_month_start.astype(int)
            df[f"{col}_es_fin_mes"] = dt.dt.is_month_end.astype(int)

            if self.codificacion_ciclica:
                df[f"{col}_mes_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
                df[f"{col}_mes_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
                df[f"{col}_dsem_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
                df[f"{col}_dsem_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
                df[f"{col}_hora_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
                df[f"{col}_hora_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)

            if self.eliminar_original:
                df = df.drop(columns=[col])

        return df


# ──────────────────────────────────────────────────────────────────────────────
# GENERADOR DE CARACTERÍSTICAS POLINOMIALES
# ──────────────────────────────────────────────────────────────────────────────

class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Genera características polinomiales e interacciones para columnas numéricas.

    Envuelve PolynomialFeatures de sklearn con interfaz compatible con pandas.
    """

    def __init__(
        self,
        columnas: Optional[List[str]] = None,
        grado: int = 2,
        solo_interacciones: bool = False,
        incluir_sesgo: bool = False,
    ):
        self.columnas = columnas
        self.grado = grado
        self.solo_interacciones = solo_interacciones
        self.incluir_sesgo = incluir_sesgo
        self._poly = None
        self._nombres_caracteristicas: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "PolynomialFeatureGenerator":
        cols = self.columnas or X.select_dtypes(include=np.number).columns.tolist()
        self._cols = cols
        self._poly = PolynomialFeatures(
            degree=self.grado,
            interaction_only=self.solo_interacciones,
            include_bias=self.incluir_sesgo,
        )
        self._poly.fit(X[cols])
        self._nombres_caracteristicas = self._poly.get_feature_names_out(cols).tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arreglo_poly = self._poly.transform(X[self._cols])
        df_poly = pd.DataFrame(arreglo_poly, columns=self._nombres_caracteristicas, index=X.index)
        cols_originales = [c for c in X.columns if c not in self._cols]
        return pd.concat([X[cols_originales].reset_index(drop=True),
                          df_poly.reset_index(drop=True)], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# CODIFICADOR POR OBJETIVO (TARGET ENCODING)
# ──────────────────────────────────────────────────────────────────────────────

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Codificación por media objetivo para variables categóricas con suavizado
    para evitar sobreajuste en categorías poco frecuentes.

    Fórmula (suavizada):
        codificado = (n * media_categoria + m * media_global) / (n + m)
    donde m es el factor de suavizado (min_samples_leaf).
    """

    def __init__(
        self,
        columnas: Optional[List[str]] = None,
        suavizado: float = 10.0,
    ):
        self.columnas = columnas
        self.suavizado = suavizado
        self._codificaciones: Dict[str, Dict[Any, float]] = {}
        self._media_global: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        cols = self.columnas or X.select_dtypes(include=["object", "category"]).columns.tolist()
        self._cols = cols
        self._media_global = float(y.mean())
        for col in cols:
            df_tmp = pd.DataFrame({"cat": X[col], "objetivo": y})
            stats = df_tmp.groupby("cat")["objetivo"].agg(["count", "mean"])
            suav = (
                (stats["count"] * stats["mean"] + self.suavizado * self._media_global)
                / (stats["count"] + self.suavizado)
            )
            self._codificaciones[col] = suav.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self._cols:
            df[f"{col}_enc_objetivo"] = df[col].map(self._codificaciones[col]).fillna(self._media_global)
        return df


# ──────────────────────────────────────────────────────────────────────────────
# CODIFICADOR POR FRECUENCIA
# ──────────────────────────────────────────────────────────────────────────────

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Codifica variables categóricas por su frecuencia (proporción) en el conjunto de entrenamiento.
    Útil cuando la cardinalidad es alta y el orden no importa.
    """

    def __init__(self, columnas: Optional[List[str]] = None, normalizar: bool = True):
        self.columnas = columnas
        self.normalizar = normalizar
        self._mapas_frecuencia: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        cols = self.columnas or X.select_dtypes(include=["object", "category"]).columns.tolist()
        self._cols = cols
        for col in cols:
            frec = X[col].value_counts(normalize=self.normalizar)
            self._mapas_frecuencia[col] = frec.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self._cols:
            df[f"{col}_enc_frec"] = df[col].map(self._mapas_frecuencia[col]).fillna(0)
        return df


# ──────────────────────────────────────────────────────────────────────────────
# DISCRETIZADOR (BINNING)
# ──────────────────────────────────────────────────────────────────────────────

class Binner(BaseEstimator, TransformerMixin):
    """
    Discretiza características numéricas continuas en intervalos.
    Soporta ancho igual, cuantiles (frecuencia igual) e intervalos personalizados.
    """

    def __init__(
        self,
        columnas: Optional[List[str]] = None,
        n_bins: int = 5,
        estrategia: str = "quantile",
        etiquetas: Optional[List[str]] = None,
    ):
        self.columnas = columnas
        self.n_bins = n_bins
        self.estrategia = estrategia
        self.etiquetas = etiquetas
        self._bordes_bins: Dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "Binner":
        cols = self.columnas or X.select_dtypes(include=np.number).columns.tolist()
        self._cols = cols
        for col in cols:
            if self.estrategia == "quantile":
                cuantiles = np.linspace(0, 100, self.n_bins + 1)
                bordes = np.percentile(X[col].dropna(), cuantiles)
            elif self.estrategia == "uniform":
                bordes = np.linspace(X[col].min(), X[col].max(), self.n_bins + 1)
            else:
                raise ValueError("La estrategia debe ser 'quantile' o 'uniform'.")
            self._bordes_bins[col] = np.unique(bordes)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self._cols:
            df[f"{col}_bin"] = pd.cut(
                df[col],
                bins=self._bordes_bins[col],
                labels=self.etiquetas,
                include_lowest=True,
                duplicates="drop",
            )
        return df


# ──────────────────────────────────────────────────────────────────────────────
# SELECTOR DE CARACTERÍSTICAS
# ──────────────────────────────────────────────────────────────────────────────

class StatisticalFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selecciona las k mejores características usando funciones de puntuación estadística.

    Soporta:
        - f_classif       : Prueba F ANOVA (clasificación)
        - f_regression    : Prueba F (regresión)
        - mutual_info     : Información mutua (clasificación)
    """

    FUNCIONES_PUNTUACION = {
        "f_classif": f_classif,
        "f_regression": f_regression,
        "mutual_info": mutual_info_classif,
    }

    def __init__(
        self,
        funcion_puntuacion: str = "f_classif",
        k: Union[int, str] = 10,
    ):
        if funcion_puntuacion not in self.FUNCIONES_PUNTUACION:
            raise ValueError(f"funcion_puntuacion debe ser una de {list(self.FUNCIONES_PUNTUACION.keys())}")
        self.funcion_puntuacion = funcion_puntuacion
        self.k = k
        self._selector = None
        self._columnas_seleccionadas: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StatisticalFeatureSelector":
        X_numerico = X.select_dtypes(include=np.number)
        self._todas_cols_numericas = X_numerico.columns.tolist()
        self._selector = SelectKBest(
            score_func=self.FUNCIONES_PUNTUACION[self.funcion_puntuacion],
            k=min(self.k, len(self._todas_cols_numericas)) if isinstance(self.k, int) else self.k,
        )
        self._selector.fit(X_numerico.fillna(0), y)
        mascara = self._selector.get_support()
        self._columnas_seleccionadas = [c for c, s in zip(self._todas_cols_numericas, mascara) if s]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        no_numericas = X.select_dtypes(exclude=np.number)
        return pd.concat([no_numericas, X[self._columnas_seleccionadas]], axis=1)

    def puntuaciones_caracteristicas(self) -> pd.DataFrame:
        """Retorna las importancias de características como DataFrame ordenado."""
        puntuaciones = self._selector.scores_
        return (
            pd.DataFrame({"caracteristica": self._todas_cols_numericas, "puntuacion": puntuaciones})
            .sort_values("puntuacion", ascending=False)
            .reset_index(drop=True)
        )


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ──────────────────────────────────────────────────────────────────────────────

class FeatureEngineeringPipeline:
    """
    Orquesta todos los pasos de ingeniería de características:
      1. Extracción de características datetime
      2. Codificación por frecuencia para categóricas
      3. Codificación por objetivo (si se proporciona y)
      4. Características polinomiales
      5. Discretización (binning)
      6. Selección de características
    """

    def __init__(
        self,
        datetime_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        poly_degree: int = 2,
        n_bins: int = 5,
        k_best: int = 20,
        task: str = "classification",
    ):
        self.datetime_cols = datetime_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.poly_degree = poly_degree
        self.n_bins = n_bins
        self.k_best = k_best
        self.task = task
        self._pasos: List[Any] = []

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Ajusta y transforma el conjunto de datos a través de todos los pasos de ingeniería."""
        df = X.copy()

        # 1. DateTime
        if self.datetime_cols:
            extractor_dt = DateTimeFeatureExtractor(columnas=self.datetime_cols, eliminar_original=True)
            df = extractor_dt.fit_transform(df)
            self._pasos.append(("datetime", extractor_dt))

        # 2. Codificación por frecuencia
        enc_frec = FrequencyEncoder(columnas=self.cat_cols)
        df = enc_frec.fit_transform(df)
        self._pasos.append(("enc_frec", enc_frec))

        # 3. Codificación por objetivo (solo si se proporciona y)
        if y is not None and self.cat_cols:
            enc_objetivo = TargetEncoder(columnas=self.cat_cols)
            df = enc_objetivo.fit(df, y).transform(df)
            self._pasos.append(("enc_objetivo", enc_objetivo))

        # 4. Discretización
        discretizador = Binner(columnas=self.num_cols, n_bins=self.n_bins)
        df = discretizador.fit_transform(df)
        self._pasos.append(("discretizador", discretizador))

        # 5. Selección de características
        if y is not None:
            func_punt = "f_classif" if self.task == "classification" else "f_regression"
            selector = StatisticalFeatureSelector(funcion_puntuacion=func_punt, k=self.k_best)
            df = selector.fit_transform(df, y)
            self._pasos.append(("selector", selector))

        return df

    def get_feature_scores(self) -> Optional[pd.DataFrame]:
        """Recupera las puntuaciones del paso de selección (si se aplicó)."""
        for nombre, paso in self._pasos:
            if nombre == "selector":
                return paso.puntuaciones_caracteristicas()
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Demostración
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    df = pd.DataFrame(
        {
            "fecha_registro": pd.date_range("2022-01-01", periods=n, freq="D"),
            "categoria": np.random.choice(["A", "B", "C", "D"], n),
            "region": np.random.choice(["Norte", "Sur", "Este", "Oeste"], n),
            "edad": np.random.randint(18, 70, n),
            "ingresos": np.random.normal(50000, 15000, n),
            "puntaje": np.random.uniform(0, 100, n),
        }
    )
    y = pd.Series(np.random.randint(0, 2, n), name="churn")

    pipeline = FeatureEngineeringPipeline(
        datetime_cols=["fecha_registro"],
        cat_cols=["categoria", "region"],
        num_cols=["edad", "ingresos", "puntaje"],
        poly_degree=2,
        n_bins=5,
        k_best=15,
        task="classification",
    )
    df_ingenieria = pipeline.fit_transform(df, y)
    print(f"Forma original:    {df.shape}")
    print(f"Forma con ingeniería: {df_ingenieria.shape}")
    print("\nTop 10 características por puntuación F:")
    print(pipeline.get_feature_scores().head(10))
