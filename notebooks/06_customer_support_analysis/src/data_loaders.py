"""
📥 Data Loaders — Implementaciones concretas de IDataLoader
=============================================================
Principio de Responsabilidad Única (SRP):
  Cada loader se ocupa SOLO de cargar datos de su fuente.

Principio Abierto/Cerrado (OCP):
  Para agregar una nueva fuente (API, Parquet, SQL), solo se crea
  una nueva clase que implementa IDataLoader. No se modifica nada existente.

Autor: Dody Dueñas
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

from interfaces import IDataLoader

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  CSVDataLoader — Carga desde archivo CSV
# ══════════════════════════════════════════════════════════════
class CSVDataLoader(IDataLoader):
    """
    Carga datos desde un archivo CSV.

    Parámetros
    ----------
    file_path : str
        Ruta al archivo CSV.
    required_columns : list
        Columnas mínimas que debe tener el archivo.
    nrows : int, opcional
        Número de filas a leer (útil para pruebas rápidas).
    encoding : str
        Codificación del archivo (default: 'utf-8').
    """

    REQUIRED_COLUMNS: List[str] = [
        'conv_id', 'turn_index', 'role', 'text', 'timestamp',
        'industry', 'product', 'issue_type', 'language', 'channel',
        'customer_name', 'agent_name', 'overall_sentiment',
        'overall_urgency', 'outcome', 'primary_intent'
    ]

    def __init__(
        self,
        file_path: str,
        required_columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        encoding: str = 'utf-8'
    ):
        self.file_path = Path(file_path)
        self.required_columns = required_columns or self.REQUIRED_COLUMNS
        self.nrows = nrows
        self.encoding = encoding

    def load(self) -> pd.DataFrame:
        """Carga el CSV y aplica tipos de datos básicos."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.file_path}")

        logger.info(f"Cargando: {self.file_path.name}")
        df = pd.read_csv(
            self.file_path,
            nrows=self.nrows,
            encoding=self.encoding,
            parse_dates=['timestamp'],
            infer_datetime_format=True
        )
        logger.info(f"Cargadas {len(df):,} filas × {df.shape[1]} columnas.")
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Verifica que todas las columnas requeridas estén presentes."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
        logger.info("Validación de columnas: ✅ OK")
        return True


# ══════════════════════════════════════════════════════════════
#  ParquetDataLoader — Carga desde Parquet (OCP: nueva fuente)
# ══════════════════════════════════════════════════════════════
class ParquetDataLoader(IDataLoader):
    """
    Carga datos desde un archivo Parquet.
    Demuestra el Principio Abierto/Cerrado: nueva fuente de datos
    sin modificar CSVDataLoader ni ningún otro módulo.
    """

    def __init__(self, file_path: str, required_columns: Optional[List[str]] = None):
        self.file_path = Path(file_path)
        self.required_columns = required_columns or []

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.file_path}")
        logger.info(f"Cargando Parquet: {self.file_path.name}")
        return pd.read_parquet(self.file_path)

    def validate(self, df: pd.DataFrame) -> bool:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
        return True
