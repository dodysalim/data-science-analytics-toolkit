from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

class IObserver(ABC):
    """
    Observer Pattern: Interface for logging and tracking progress.
    """
    @abstractmethod
    def update(self, step_name: str, status: str, message: str) -> None:
        pass

class IPipelineStep(ABC):
    """
    Strategy/Command Pattern: Interface for any analytical module in the pipeline.
    Complies with Open/Closed Principle (OCP) and Liskov Substitution Principle (LSP).
    """
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta la logica del modulo.
        :param data: DataFrame original.
        :param context: Diccionario de artefactos compartidos entre pasos.
        :return: Contexto actualizado.
        """
        pass
