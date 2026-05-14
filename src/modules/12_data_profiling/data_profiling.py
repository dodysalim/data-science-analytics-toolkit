"""
Generador de Perfiles de Datos y Reporte de Calidad
======================================================
Perfilado automático de conjuntos de datos que genera reportes de calidad:
- Estadísticas a nivel de columna (numérica, categórica, datetime)
- Análisis de valores faltantes
- Detección de duplicados
- Análisis de cardinalidad y unicidad
- Clasificación de la forma de distribución
- Inferencia de tipos y recomendaciones de conversión
- Puntuación de calidad general de datos

Autor: Dody Dueñas
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# PERFILADORES POR TIPO DE COLUMNA
# ──────────────────────────────────────────────────────────────────────────────

class NumericProfiler:
    """Genera perfil estadístico para una columna numérica."""

    def profile(self, serie: pd.Series) -> Dict[str, Any]:
        """
        Calcula estadísticas descriptivas para una serie numérica.

        Retorna
        -------
        dict con: conteo, faltantes, pct_faltantes, media, mediana, desv_std,
                  min, max, q1, q3, iqr, asimetria, curtosis, ceros, negativos,
                  es_entero, forma_distribucion, valores_atipicos_iqr
        """
        limpio = serie.dropna()
        n_total = len(serie)
        n_faltantes = serie.isna().sum()

        q1 = float(limpio.quantile(0.25))
        q3 = float(limpio.quantile(0.75))
        asimetria = float(limpio.skew()) if len(limpio) > 3 else np.nan
        curtosis = float(limpio.kurt()) if len(limpio) > 3 else np.nan

        # Heurística de forma de distribución
        if abs(asimetria) < 0.5:
            forma = "aproximadamente_normal"
        elif asimetria > 1.5:
            forma = "muy_sesgada_derecha"
        elif asimetria > 0.5:
            forma = "sesgada_derecha"
        elif asimetria < -1.5:
            forma = "muy_sesgada_izquierda"
        else:
            forma = "sesgada_izquierda"

        return {
            "tipo_dato": str(serie.dtype),
            "conteo": int(len(limpio)),
            "faltantes": int(n_faltantes),
            "pct_faltantes": round(n_faltantes / n_total * 100, 2),
            "media": round(float(limpio.mean()), 6),
            "mediana": round(float(limpio.median()), 6),
            "moda": round(float(limpio.mode().iloc[0]), 6) if len(limpio) > 0 else None,
            "desv_std": round(float(limpio.std()), 6),
            "varianza": round(float(limpio.var()), 6),
            "min": round(float(limpio.min()), 6),
            "max": round(float(limpio.max()), 6),
            "rango": round(float(limpio.max() - limpio.min()), 6),
            "q1": round(q1, 6),
            "q3": round(q3, 6),
            "iqr": round(q3 - q1, 6),
            "p5": round(float(limpio.quantile(0.05)), 6),
            "p95": round(float(limpio.quantile(0.95)), 6),
            "asimetria": round(asimetria, 4),
            "curtosis": round(curtosis, 4),
            "ceros": int((limpio == 0).sum()),
            "pct_ceros": round((limpio == 0).sum() / n_total * 100, 2),
            "negativos": int((limpio < 0).sum()),
            "es_valor_entero": bool(limpio.dropna().apply(lambda x: x == int(x)).all()),
            "forma_distribucion": forma,
            "atipicos_iqr": int(((limpio < q1 - 1.5 * (q3 - q1)) | (limpio > q3 + 1.5 * (q3 - q1))).sum()),
        }


class CategoricalProfiler:
    """Genera perfil estadístico para una columna categórica u objeto."""

    def profile(self, serie: pd.Series, top_n: int = 10) -> Dict[str, Any]:
        """
        Calcula estadísticas para una serie categórica.

        Retorna
        -------
        dict con: conteo, faltantes, pct_faltantes, n_unicos, razon_cardinalidad,
                  valores_top, frecuencias_top, es_binario, es_identificador
        """
        n_total = len(serie)
        n_faltantes = serie.isna().sum()
        limpio = serie.dropna()
        n_unicos = int(limpio.nunique())
        conteos_valores = limpio.value_counts()
        top = conteos_valores.head(top_n)

        return {
            "tipo_dato": str(serie.dtype),
            "conteo": int(len(limpio)),
            "faltantes": int(n_faltantes),
            "pct_faltantes": round(n_faltantes / n_total * 100, 2),
            "n_unicos": n_unicos,
            "razon_cardinalidad": round(n_unicos / len(limpio) * 100, 2) if len(limpio) > 0 else 0,
            "valores_top": top.index.tolist(),
            "frecuencias_top": top.values.tolist(),
            "proporciones_top": (top / len(limpio)).round(4).values.tolist(),
            "mas_frecuente": str(conteos_valores.index[0]) if len(conteos_valores) > 0 else None,
            "pct_mas_frecuente": round(conteos_valores.iloc[0] / n_total * 100, 2) if len(conteos_valores) > 0 else 0,
            "es_binario": n_unicos == 2,
            "es_posible_identificador": n_unicos / len(limpio) > 0.95 if len(limpio) > 0 else False,
            "es_constante": n_unicos == 1,
            "entropia": round(float(stats.entropy(conteos_valores / len(limpio), base=2)), 4) if len(conteos_valores) > 0 else 0,
        }


class DatetimeProfiler:
    """Genera perfil estadístico para una columna datetime."""

    def profile(self, serie: pd.Series) -> Dict[str, Any]:
        """Calcula estadísticas temporales."""
        try:
            dt = pd.to_datetime(serie, errors="coerce")
        except Exception:
            return {"error": "No se pudo interpretar como datetime"}

        n_total = len(dt)
        n_faltantes = dt.isna().sum()
        limpio = dt.dropna()

        return {
            "tipo_dato": str(serie.dtype),
            "conteo": int(len(limpio)),
            "faltantes": int(n_faltantes),
            "pct_faltantes": round(n_faltantes / n_total * 100, 2),
            "fecha_min": str(limpio.min()),
            "fecha_max": str(limpio.max()),
            "rango_dias": int((limpio.max() - limpio.min()).days) if len(limpio) > 1 else 0,
            "n_fechas_unicas": int(limpio.dt.date.nunique()),
            "año_mas_comun": int(limpio.dt.year.mode().iloc[0]) if len(limpio) > 0 else None,
            "mes_mas_comun": int(limpio.dt.month.mode().iloc[0]) if len(limpio) > 0 else None,
            "dia_semana_mas_comun": int(limpio.dt.dayofweek.mode().iloc[0]) if len(limpio) > 0 else None,
            "tiene_componente_hora": bool((limpio.dt.hour != 0).any()),
        }


# ──────────────────────────────────────────────────────────────────────────────
# PERFILADOR DE CONJUNTO DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

class DatasetProfiler:
    """
    Orquesta el perfilado completo del conjunto de datos para todos los tipos de columna.

    Retorna un reporte de perfil estructurado con:
    - Estadísticas por columna
    - Datos de mapa de calor de valores faltantes
    - Análisis de duplicados
    - Puntuación de calidad de datos
    """

    def __init__(self, columnas_datetime: Optional[List[str]] = None):
        self.columnas_datetime = columnas_datetime or []
        self._perfilador_numerico = NumericProfiler()
        self._perfilador_cat = CategoricalProfiler()
        self._perfilador_dt = DatetimeProfiler()

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta el perfilado completo sobre un DataFrame.

        Retorna
        -------
        dict con: 'resumen', 'columnas', 'faltantes', 'duplicados', 'puntuacion_calidad'
        """
        reporte = {}

        # ── Resumen general ───────────────────────────────────────────────────
        reporte["resumen"] = {
            "n_filas": len(df),
            "n_columnas": len(df.columns),
            "n_celdas": len(df) * len(df.columns),
            "memoria_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
            "n_cols_numericas": len(df.select_dtypes(include=np.number).columns),
            "n_cols_categoricas": len(df.select_dtypes(include=["object", "category"]).columns),
            "n_cols_datetime": len(df.select_dtypes(include=["datetime64"]).columns),
            "total_faltantes": int(df.isna().sum().sum()),
            "pct_faltantes_total": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        }

        # ── Perfiles por columna ──────────────────────────────────────────────
        columnas = {}
        for col in df.columns:
            if col in self.columnas_datetime or pd.api.types.is_datetime64_any_dtype(df[col]):
                columnas[col] = {"tipo": "datetime", **self._perfilador_dt.profile(df[col])}
            elif pd.api.types.is_numeric_dtype(df[col]):
                columnas[col] = {"tipo": "numerico", **self._perfilador_numerico.profile(df[col])}
            else:
                columnas[col] = {"tipo": "categorico", **self._perfilador_cat.profile(df[col])}
        reporte["columnas"] = columnas

        # ── Análisis de faltantes ─────────────────────────────────────────────
        serie_faltantes = df.isna().sum()
        pct_faltantes = (df.isna().sum() / len(df) * 100).round(2)
        reporte["faltantes"] = pd.DataFrame({
            "conteo_faltantes": serie_faltantes,
            "pct_faltantes": pct_faltantes,
        }).sort_values("pct_faltantes", ascending=False).to_dict()

        # ── Duplicados ────────────────────────────────────────────────────────
        n_dupes = int(df.duplicated().sum())
        reporte["duplicados"] = {
            "n_filas_duplicadas": n_dupes,
            "pct_duplicados": round(n_dupes / len(df) * 100, 2),
            "es_preocupante": n_dupes > 0,
        }

        # ── Puntuación de calidad ─────────────────────────────────────────────
        reporte["puntuacion_calidad"] = self._calcular_puntuacion_calidad(df, reporte)

        return reporte

    def _calcular_puntuacion_calidad(self, df: pd.DataFrame, reporte: Dict) -> Dict[str, Any]:
        """
        Calcula una puntuación de calidad de datos de 0-100 basada en:
        - Completitud (penalización por valores faltantes)
        - Unicidad (penalización por filas duplicadas)
        - Consistencia (penalización por columnas constantes)
        - Validez (columnas identificadoras marcadas como categóricas)
        """
        completitud = 100 - reporte["resumen"]["pct_faltantes_total"]
        unicidad = 100 - reporte["duplicados"]["pct_duplicados"]

        n_constantes = sum(
            1 for v in reporte["columnas"].values()
            if v.get("es_constante", False)
        )
        n_identificadores = sum(
            1 for v in reporte["columnas"].values()
            if v.get("es_posible_identificador", False)
        )
        consistencia = max(0, 100 - n_constantes * 10)
        validez = max(0, 100 - n_identificadores * 5)

        total = round((completitud + unicidad + consistencia + validez) / 4, 1)
        calificacion = "A" if total >= 90 else "B" if total >= 75 else "C" if total >= 60 else "D"

        return {
            "completitud": round(completitud, 2),
            "unicidad": round(unicidad, 2),
            "consistencia": round(consistencia, 2),
            "validez": round(validez, 2),
            "total": total,
            "calificacion": calificacion,
        }

    def summary_table(self, reporte: Dict) -> pd.DataFrame:
        """Retorna una tabla resumen ordenada de todos los perfiles de columna."""
        filas = []
        for col, perf in reporte["columnas"].items():
            fila = {"columna": col, "tipo": perf.get("tipo")}
            fila["pct_faltantes"] = perf.get("pct_faltantes")
            fila["n_unicos"] = perf.get("n_unicos") or perf.get("n_fechas_unicas")
            if perf["tipo"] == "numerico":
                fila["media"] = perf.get("media")
                fila["desv_std"] = perf.get("desv_std")
                fila["min"] = perf.get("min")
                fila["max"] = perf.get("max")
                fila["asimetria"] = perf.get("asimetria")
                fila["atipicos"] = perf.get("atipicos_iqr")
                fila["distribucion"] = perf.get("forma_distribucion")
            elif perf["tipo"] == "categorico":
                fila["mas_frecuente"] = perf.get("mas_frecuente")
                fila["pct_mas_frecuente"] = perf.get("pct_mas_frecuente")
                fila["razon_cardinalidad"] = perf.get("razon_cardinalidad")
            filas.append(fila)
        return pd.DataFrame(filas).set_index("columna")

    def export_json(self, reporte: Dict, ruta: str) -> None:
        """Exporta el reporte completo a un archivo JSON."""
        def convertir(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(reporte, f, indent=2, default=convertir, ensure_ascii=False)

    def print_summary(self, reporte: Dict) -> None:
        """Imprime un resumen formateado en la consola."""
        res = reporte["resumen"]
        pc = reporte["puntuacion_calidad"]
        print("=" * 60)
        print("  REPORTE DE PERFILADO DE DATOS")
        print("=" * 60)
        print(f"  Filas:              {res['n_filas']:,}")
        print(f"  Columnas:           {res['n_columnas']}")
        print(f"  Memoria:            {res['memoria_mb']} MB")
        print(f"  Total Faltantes:    {res['total_faltantes']:,} ({res['pct_faltantes_total']}%)")
        print(f"  Filas Duplicadas:   {reporte['duplicados']['n_filas_duplicadas']:,}")
        print("-" * 60)
        print("  PUNTUACIÓN DE CALIDAD DE DATOS")
        print(f"  Completitud:   {pc['completitud']}%")
        print(f"  Unicidad:      {pc['unicidad']}%")
        print(f"  Consistencia:  {pc['consistencia']}%")
        print(f"  Total:         {pc['total']} / 100  [Calificación: {pc['calificacion']}]")
        print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Demostración
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "id_cliente": [f"C{i:05d}" for i in range(n)],
        "edad": np.random.randint(18, 80, n).astype(float),
        "ingresos": np.random.lognormal(10, 0.8, n),
        "puntaje": np.random.beta(2, 5, n) * 100,
        "segmento": np.random.choice(["Premium", "Estándar", "Básico", "Prueba"], n),
        "pais": np.random.choice(["Colombia", "México", "Argentina", "Chile", "Perú"], n),
        "fecha_registro": pd.date_range("2020-01-01", periods=n, freq="H"),
        "activo": np.random.choice([True, False], n),
    })

    # Introducir 5% de valores faltantes
    for col in ["edad", "ingresos", "puntaje", "pais"]:
        df.loc[df.sample(frac=0.05).index, col] = np.nan

    # Agregar algunos duplicados
    df = pd.concat([df, df.sample(20)], ignore_index=True)

    perfilador = DatasetProfiler()
    reporte = perfilador.profile(df)
    perfilador.print_summary(reporte)

    print("\n=== Tabla Resumen de Columnas ===")
    print(perfilador.summary_table(reporte).to_string())
