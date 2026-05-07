"""
🧹 Data Cleaning Utilities
===========================
Librería de funciones reutilizables para limpieza y preprocesamiento de datos.

Incluye:
- Limpieza de valores nulos con estrategias inteligentes
- Normalización y estandarización
- Encoding de variables categóricas
- Detección y remoción de duplicados
- Formateo de fechas y texto
- Validación de datos

Autor: Dody Dueñas
Fecha: 2026
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from typing import Union, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
#  MÓDULO 1: Manejo de Valores Nulos
# ══════════════════════════════════════════════════════════════

def report_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un reporte completo de valores nulos por columna.

    Returns
    -------
    pd.DataFrame con columnas: count, percentage, dtype
    """
    null_count = df.isnull().sum()
    null_pct = (null_count / len(df) * 100).round(2)
    report = pd.DataFrame({
        'null_count': null_count,
        'null_percentage': null_pct,
        'dtype': df.dtypes
    })
    return report[report['null_count'] > 0].sort_values('null_percentage', ascending=False)


def fill_nulls_smart(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
    """
    Rellena valores nulos usando una estrategia inteligente.

    Parámetros
    ----------
    df : pd.DataFrame
    strategy : str
        'auto'   → Media para numéricas, moda para categóricas
        'median' → Mediana para numéricas
        'mode'   → Moda para todas
        'zero'   → 0 para numéricas, 'Unknown' para categóricas
        'drop'   → Elimina filas con nulos
    """
    df = df.copy()

    if strategy == 'drop':
        return df.dropna()

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'zero':
                df[col].fillna(0, inplace=True)
            else:  # auto o mode
                df[col].fillna(df[col].mean(), inplace=True)
        else:
            if strategy == 'zero':
                df[col].fillna('Unknown', inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

    print(f"✅ Valores nulos tratados con estrategia '{strategy}'.")
    return df


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Elimina columnas con más del `threshold` (%) de valores nulos.

    Parámetros
    ----------
    threshold : float
        Porcentaje (0.0 - 1.0). Por defecto 0.5 (50%).
    """
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()
    if cols_to_drop:
        print(f"🗑️  Columnas eliminadas por alta nulidad (>{threshold*100:.0f}%): {cols_to_drop}")
    return df.drop(columns=cols_to_drop)


# ══════════════════════════════════════════════════════════════
#  MÓDULO 2: Duplicados
# ══════════════════════════════════════════════════════════════

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None, keep: str = 'first') -> pd.DataFrame:
    """
    Elimina filas duplicadas y reporta cuántas se removieron.

    Parámetros
    ----------
    subset : list, opcional
        Columnas a considerar para duplicados. Si None, usa todas.
    keep : str
        'first', 'last', o False (elimina todos los duplicados).
    """
    original_len = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    removed = original_len - len(df)
    print(f"🗑️  {removed:,} duplicados eliminados ({removed/original_len*100:.2f}% del total).")
    return df


# ══════════════════════════════════════════════════════════════
#  MÓDULO 3: Normalización y Escalado
# ══════════════════════════════════════════════════════════════

def normalize_columns(df: pd.DataFrame, cols: List[str] = None, method: str = 'minmax') -> pd.DataFrame:
    """
    Normaliza o estandariza columnas numéricas.

    Parámetros
    ----------
    cols : list, opcional
        Columnas a normalizar. Si None, usa todas las numéricas.
    method : str
        'minmax'   → Escala a [0, 1]
        'standard' → Media 0, desviación estándar 1
        'log'      → Transformación logarítmica (log1p)
    """
    df = df.copy()
    numeric_cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()

    if method == 'minmax':
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif method == 'standard':
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif method == 'log':
        for col in numeric_cols:
            if (df[col] >= 0).all():
                df[col] = np.log1p(df[col])
            else:
                print(f"⚠️  '{col}' tiene valores negativos, se omite log.")

    print(f"✅ Normalización '{method}' aplicada a: {numeric_cols}")
    return df


# ══════════════════════════════════════════════════════════════
#  MÓDULO 4: Encoding de Variables Categóricas
# ══════════════════════════════════════════════════════════════

def encode_categoricals(df: pd.DataFrame, cols: List[str] = None, method: str = 'onehot') -> pd.DataFrame:
    """
    Codifica variables categóricas.

    Parámetros
    ----------
    cols : list, opcional
        Columnas a codificar. Si None, usa todas las categóricas.
    method : str
        'onehot'  → One-Hot Encoding (pd.get_dummies)
        'label'   → Label Encoding (0, 1, 2, ...)
        'ordinal' → Igual que label
    """
    df = df.copy()
    cat_cols = cols or df.select_dtypes(include=['object', 'category']).columns.tolist()

    if method == 'onehot':
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        print(f"✅ One-Hot Encoding aplicado. Nuevas dimensiones: {df.shape}")
    elif method in ('label', 'ordinal'):
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        print(f"✅ Label Encoding aplicado a: {cat_cols}")

    return df


# ══════════════════════════════════════════════════════════════
#  MÓDULO 5: Limpieza de Texto
# ══════════════════════════════════════════════════════════════

def clean_text_column(series: pd.Series, operations: List[str] = None) -> pd.Series:
    """
    Limpia una columna de texto aplicando operaciones en cadena.

    Operaciones disponibles:
    - 'lowercase'    → Convierte a minúsculas
    - 'strip'        → Elimina espacios al inicio/fin
    - 'remove_punct' → Elimina puntuación
    - 'remove_digits'→ Elimina dígitos
    - 'remove_extra_spaces' → Colapsa espacios múltiples
    - 'remove_special' → Elimina caracteres especiales

    Ejemplo
    -------
    df['nombre'] = clean_text_column(df['nombre'], ['lowercase', 'strip', 'remove_extra_spaces'])
    """
    ops = operations or ['lowercase', 'strip', 'remove_extra_spaces']
    s = series.astype(str).copy()

    for op in ops:
        if op == 'lowercase':
            s = s.str.lower()
        elif op == 'strip':
            s = s.str.strip()
        elif op == 'remove_punct':
            s = s.apply(lambda x: re.sub(r'[^\w\s]', '', x))
        elif op == 'remove_digits':
            s = s.apply(lambda x: re.sub(r'\d+', '', x))
        elif op == 'remove_extra_spaces':
            s = s.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        elif op == 'remove_special':
            s = s.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

    return s


# ══════════════════════════════════════════════════════════════
#  MÓDULO 6: Manejo de Fechas
# ══════════════════════════════════════════════════════════════

def parse_dates(df: pd.DataFrame, date_cols: List[str], format: str = None) -> pd.DataFrame:
    """
    Convierte columnas de texto a datetime y extrae características útiles.

    Genera columnas adicionales: _year, _month, _day, _dayofweek, _quarter
    """
    df = df.copy()
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], format=format, infer_datetime_format=True, errors='coerce')
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
            print(f"✅ Fechas procesadas para '{col}'.")
        except Exception as e:
            print(f"⚠️  Error en '{col}': {e}")
    return df


# ══════════════════════════════════════════════════════════════
#  MÓDULO 7: Detección y Remoción de Outliers
# ══════════════════════════════════════════════════════════════

def remove_outliers_iqr(df: pd.DataFrame, cols: List[str] = None, factor: float = 1.5) -> pd.DataFrame:
    """
    Elimina filas con outliers según el método IQR.

    Parámetros
    ----------
    cols : list, opcional
        Columnas a analizar. Si None, usa todas las numéricas.
    factor : float
        Factor de IQR. Por defecto 1.5 (estándar). Use 3.0 para outliers extremos.
    """
    df = df.copy()
    numeric_cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
    original_len = len(df)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    removed = original_len - len(df)
    print(f"🗑️  {removed:,} outliers eliminados ({removed/original_len*100:.2f}% del total).")
    return df


def cap_outliers(df: pd.DataFrame, cols: List[str] = None, factor: float = 1.5) -> pd.DataFrame:
    """
    En lugar de eliminar outliers, los recorta (capping/winsorizing) a los límites del IQR.
    """
    df = df.copy()
    numeric_cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    print(f"✅ Outliers recortados (capping) en: {numeric_cols}")
    return df


# ══════════════════════════════════════════════════════════════
#  MÓDULO 8: Pipeline Completo de Limpieza
# ══════════════════════════════════════════════════════════════

def full_cleaning_pipeline(
    df: pd.DataFrame,
    null_strategy: str = 'auto',
    null_threshold: float = 0.5,
    encoding_method: str = 'label',
    normalize_method: str = 'standard',
    remove_dups: bool = True,
    remove_outliers: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Pipeline completo de limpieza de datos.

    Pasos:
    1. Eliminar columnas con alta nulidad
    2. Rellenar valores nulos
    3. Eliminar duplicados
    4. Codificar categóricas
    5. Normalizar numéricas
    6. Remover outliers (opcional)

    Returns
    -------
    pd.DataFrame limpio y listo para modelado.
    """
    if verbose:
        print(f"\n🚀 Iniciando Pipeline de Limpieza...")
        print(f"   Shape inicial: {df.shape}")

    df = drop_high_null_columns(df, threshold=null_threshold)
    df = fill_nulls_smart(df, strategy=null_strategy)

    if remove_dups:
        df = remove_duplicates(df)

    df = encode_categoricals(df, method=encoding_method)
    df = normalize_columns(df, method=normalize_method)

    if remove_outliers:
        df = remove_outliers_iqr(df)

    if verbose:
        print(f"\n✅ Pipeline completado. Shape final: {df.shape}")

    return df


# ──────────────────────────────────────────────
#  Ejemplo de uso
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [c.replace(' (cm)', '').replace(' ', '_') for c in df.columns]
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    # Introducir nulos artificiales para demostración
    np.random.seed(42)
    mask = np.random.rand(*df.shape) < 0.05
    df_with_nulls = df.mask(mask)

    print("📊 Reporte de Nulos:")
    print(report_nulls(df_with_nulls))

    df_clean = full_cleaning_pipeline(df_with_nulls, encoding_method='label', normalize_method='minmax')
