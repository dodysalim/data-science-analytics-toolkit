from src.patterns.facade import AnalyticsFacade
from src.patterns.interfaces import IObserver
from src.core.steps import (
    DataProfilingStep, FeatureEngineeringStep, StatisticalAnalysisStep,
    TimeSeriesStep, MLPipelineStep, NLPStep, ABTestingStep, VisualizationStep
)

class ContextInjectorStep:
    """Paso inyector para compartir el logger en el contexto."""
    def __init__(self, logger):
        self.logger = logger
        
    @property
    def name(self): return "00_ContextInit"
    
    def execute(self, data, context):
        context["logger"] = self.logger
        return context

class PipelineBuilder:
    """
    Builder Pattern: Ensambla el Facade con todos los pasos y observadores necesarios.
    """
    @staticmethod
    def build(logger: IObserver) -> AnalyticsFacade:
        facade = AnalyticsFacade()
        facade.attach_observer(logger)
        
        # Inyectamos el logger primero
        facade.add_step(ContextInjectorStep(logger))
        
        # Módulos analíticos
        facade.add_step(DataProfilingStep())
        facade.add_step(FeatureEngineeringStep())
        facade.add_step(StatisticalAnalysisStep())
        facade.add_step(TimeSeriesStep())
        facade.add_step(MLPipelineStep())
        facade.add_step(NLPStep())
        facade.add_step(ABTestingStep())
        facade.add_step(VisualizationStep())
        
        return facade
