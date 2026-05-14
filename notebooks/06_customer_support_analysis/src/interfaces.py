"""
🔷 Interfaces (Abstract Base Classes)
=======================================
Principio de Segregación de Interfaces (ISP) y
Principio de Inversión de Dependencias (DIP).

Cada interfaz define UN contrato específico y mínimo.
Los módulos de alto nivel dependen de estas abstracciones,
no de implementaciones concretas.

Autor: Dody Dueñas
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional


# ══════════════════════════════════════════════════════════════
#  IDataLoader — Contrato para carga de datos
# ══════════════════════════════════════════════════════════════
class IDataLoader(ABC):
    """
    Contrato para cualquier fuente de datos.
    Permite cambiar fácilmente entre CSV, SQL, API, Parquet, etc.
    (Principio de Inversión de Dependencias - DIP)
    """

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Carga y retorna el DataFrame crudo."""
        ...

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """Valida que el DataFrame tenga la estructura esperada."""
        ...


# ══════════════════════════════════════════════════════════════
#  IDataCleaner — Contrato para limpieza de datos
# ══════════════════════════════════════════════════════════════
class IDataCleaner(ABC):
    """
    Contrato para limpieza y preprocesamiento de datos.
    (Principio de Responsabilidad Única - SRP)
    """

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia el DataFrame y retorna la versión procesada."""
        ...


# ══════════════════════════════════════════════════════════════
#  IAnalyzer — Contrato para análisis
# ══════════════════════════════════════════════════════════════
class IAnalyzer(ABC):
    """
    Contrato para cualquier tipo de análisis.
    Nuevos análisis se agregan implementando esta interfaz
    sin modificar el código existente.
    (Principio Abierto/Cerrado - OCP)
    """

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> dict:
        """Ejecuta el análisis y retorna un diccionario de resultados."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Retorna el nombre descriptivo del análisis."""
        ...


# ══════════════════════════════════════════════════════════════
#  IReporter — Contrato para generación de reportes
# ══════════════════════════════════════════════════════════════
class IReporter(ABC):
    """
    Contrato para reportes y salidas.
    Permite agregar nuevos formatos (HTML, PDF, JSON) sin
    modificar los analizadores.
    (Principio de Inversión de Dependencias - DIP)
    """

    @abstractmethod
    def report(self, results: dict, output_path: Optional[str] = None) -> None:
        """Genera y opcionalmente guarda el reporte."""
        ...


# ══════════════════════════════════════════════════════════════
#  IVisualizer — Contrato para visualizaciones
# ══════════════════════════════════════════════════════════════
class IVisualizer(ABC):
    """
    Contrato para visualizaciones de datos.
    Cada visualización tiene su propia responsabilidad.
    (Principio de Responsabilidad Única - SRP)
    """

    @abstractmethod
    def plot(self, df: pd.DataFrame, output_dir: str = ".") -> None:
        """Genera y guarda las visualizaciones."""
        ...

    @abstractmethod
    def get_title(self) -> str:
        """Retorna el título del gráfico."""
        ...
