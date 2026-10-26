from typing import List, Any, Dict
import pandas as pd
from .interfaces import IPipelineStep, IObserver

class AnalyticsFacade:
    """
    Facade Pattern: Orquesta la ejecución de múltiples módulos analíticos
    sin exponer su complejidad al cliente.
    """
    def __init__(self):
        self._steps: List[IPipelineStep] = []
        self._observers: List[IObserver] = []

    def attach_observer(self, observer: IObserver):
        self._observers.append(observer)

    def _notify(self, step_name: str, status: str, message: str = ""):
        for obs in self._observers:
            obs.update(step_name, status, message)

    def add_step(self, step: IPipelineStep):
        """
        Agrega un paso a la pipeline cumpliendo con Open/Closed Principle.
        """
        self._steps.append(step)

    def execute_pipeline(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta todos los módulos registrados de forma secuencial,
        compartiendo un contexto (estado) entre ellos.
        """
        context = {}
        self._notify("AnalyticsFacade", "START", "Iniciando ejecución maestra")

        for step in self._steps:
            self._notify(step.name, "START")
            try:
                context = step.execute(data, context)
                self._notify(step.name, "SUCCESS", f"Módulo {step.name} completado con éxito.")
            except Exception as e:
                self._notify(step.name, "ERROR", str(e))
                # Fallback seguro para no romper toda la pipeline
            self._notify(step.name, "END")

        self._notify("AnalyticsFacade", "SUCCESS", "Todos los módulos ejecutados exitosamente")
        return context
