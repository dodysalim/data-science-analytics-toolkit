import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any

from ..patterns.interfaces import IPipelineStep

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(SRC_DIR)
OUTPUT = os.path.join(REPO_DIR, "docs")
os.makedirs(OUTPUT, exist_ok=True)

def _insert_path(module_name: str):
    path = os.path.join(SRC_DIR, "modules", module_name)
    if path not in sys.path:
        sys.path.insert(0, path)

class DataProfilingStep(IPipelineStep):
    @property
    def name(self) -> str: return "01_DataProfiling"

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        _insert_path("12_data_profiling")
        from data_profiling import DatasetProfiler
        
        perfilador = DatasetProfiler()
        reporte = perfilador.profile(data)
        
        json_path = os.path.join(OUTPUT, "reporte_perfil_solid.json")
        perfilador.export_json(reporte, json_path)
        
        context["logger"].update(self.name, "INFO", f"Reporte guardado en {json_path}")
        context["logger"].update(self.name, "INFO", f"Calidad general: {reporte['puntuacion_calidad']['calificacion']}")
        return context

class FeatureEngineeringStep(IPipelineStep):
    @property
    def name(self) -> str: return "02_FeatureEngineering"

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        _insert_path("08_feature_engineering")
        from feature_engineering import FeatureEngineeringPipeline
        
        y_objetivo = (data["outcome"] == "Resolved").astype(int)
        y_objetivo.name = "is_resolved"
        
        df_fe = data.drop(columns=["conv_id", "text", "customer_name", "agent_name", "outcome"], errors="ignore")
        
        pipe = FeatureEngineeringPipeline(
            datetime_cols=["timestamp"],
            cat_cols=["industry", "product", "issue_type", "language", "channel", "overall_sentiment", "overall_urgency", "primary_intent"],
            num_cols=["turn_index", "text_length", "response_time_mins"],
            poly_degree=1, n_bins=5, k_best=15, task="classification"
        )
        df_eng = pipe.fit_transform(df_fe, y_objetivo)
        
        context["df_eng"] = df_eng
        context["y_objetivo"] = y_objetivo
        context["logger"].update(self.name, "INFO", f"Dataset transformado: {df_eng.shape[1]} features")
        return context

class StatisticalAnalysisStep(IPipelineStep):
    @property
    def name(self) -> str: return "03_StatisticalAnalysis"

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        _insert_path("05_statistical_analysis")
        import statistical_analysis as sa
        
        # T-test
        g_a = data.loc[data["outcome"] == "Resolved", "text_length"].dropna()
        g_b = data.loc[data["outcome"] == "Escalated", "text_length"].dropna()
        t_stat, t_p = sa.ttest_ind(g_a, g_b)
        
        context["logger"].update(self.name, "INFO", f"T-test (Res vs Esc Length): p-value = {t_p:.4f}")
        return context

class TimeSeriesStep(IPipelineStep):
    @property
    def name(self) -> str: return "04_TimeSeries"

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        _insert_path("07_time_series")
        from time_series_analyzer import PronosticadorARIMA
        
        # Serie simulada
        fechas = pd.date_range("2019-01-01", periods=120, freq="MS")
        tendenc = np.linspace(100, 220, 120)
        serie_ts = pd.Series(tendenc + np.random.normal(0, 6, 120), index=fechas)
        
        pronost = PronosticadorARIMA()
        pronost.ajustar(serie_ts, orden=(1,1,1))
        f = pronost.pronosticar(pasos=3)
        
        context["logger"].update(self.name, "INFO", f"Pronóstico ARIMA a 3 meses: {list(f.values())}")
        return context

class MLPipelineStep(IPipelineStep):
    @property
    def name(self) -> str: return "05_MLPipeline"

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        _insert_path("03_ml_pipeline")
        from ml_pipeline import MLPipeline
        
        df_eng = context["df_eng"]
        y_obj = context["y_objetivo"]
        X_ml = df_eng.select_dtypes(include=np.number).fillna(0).values
        y_ml = y_obj.values
        
        ml = MLPipeline(task="classification", random_state=42)
        ml.prepare_data(X_ml, y_ml)
        resultados = ml.compare_models()
        
        context["logger"].update(self.name, "INFO", f"Entrenados {len(resultados)} modelos. Mejor modelo en ranking.")
        return context

class NLPStep(IPipelineStep):
    @property
    def name(self) -> str: return "06_NLP"

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        _insert_path("11_nlp_toolkit")
        from nlp_toolkit import NLPPipeline
        
        textos = data["text"].dropna().head(100).tolist()
        pipe = NLPPipeline(n_temas=2, max_features=200)
        res = pipe.ejecutar(textos)
        
        context["logger"].update(self.name, "INFO", f"Corpus procesado. Temas descubiertos: {len(res['temas'])}")
        return context

class ABTestingStep(IPipelineStep):
    @property
    def name(self) -> str: return "07_ABTesting"

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        _insert_path("13_ab_testing")
        from ab_testing import FrequentistABTester
        
        df_e = data[data["channel"] == "email"]
        df_i = data[data["channel"] == "in_app"]
        
        tester = FrequentistABTester()
        res = tester.proportion_test(
            (df_e["outcome"]=="Resolved").sum(), len(df_e),
            (df_i["outcome"]=="Resolved").sum(), len(df_i)
        )
        context["logger"].update(self.name, "INFO", f"Prueba Z: Significativa = {res['es_significativo']} (p={res['p_valor']:.4f})")
        return context

class VisualizationStep(IPipelineStep):
    @property
    def name(self) -> str: return "08_DataVisualization"

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        _insert_path("10_data_visualization")
        from data_visualization import CategoricalPlotter
        
        catp = CategoricalPlotter()
        fig = catp.grafico_donut(data["industry"], titulo="Industria")
        p = os.path.join(OUTPUT, "donut_industria_solid.png")
        fig.savefig(p, dpi=120); plt.close()
        
        context["logger"].update(self.name, "INFO", f"Gráficos exportados usando arquitectura SOLID en {p}")
        return context
