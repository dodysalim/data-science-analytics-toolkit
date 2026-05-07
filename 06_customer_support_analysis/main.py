"""
🚀 Main Pipeline — Customer Support Analysis
=============================================
Orquesta todo el flujo usando SOLID:

  SRP → Cada clase hace UNA cosa
  OCP → Se pueden agregar analyzers/visualizers sin tocar este archivo
  LSP → Todos los analyzers/visualizers son intercambiables
  ISP → Interfaces pequeñas y específicas
  DIP → Este archivo depende de abstracciones, no de implementaciones

Uso:
    python main.py
    python main.py --sample       # solo 5000 filas (modo rápido)
    python main.py --no-plots     # sin gráficos

Autor: Dody Dueñas
"""

import sys
import logging
import argparse
from pathlib import Path

# ─── Agregar src al path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from interfaces    import IDataLoader, IDataCleaner, IAnalyzer, IReporter, IVisualizer
from data_loaders  import CSVDataLoader
from data_cleaner  import CustomerSupportCleaner
from analyzers     import (
    SentimentAnalyzer, OutcomeAnalyzer,
    VolumeAnalyzer, UrgencyAnalyzer, IntentAnalyzer
)
from visualizers   import (
    SentimentVisualizer, OutcomeVisualizer,
    VolumeVisualizer, IntentVisualizer
)
from reporters     import ConsoleReporter

# ─── Configuración de logging ─────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─── Rutas ────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_FILE   = BASE_DIR / 'data' / 'customer_support_sample.csv'
OUTPUT_DIR  = BASE_DIR / 'output'
REPORTS_DIR = BASE_DIR / 'reports'


# ══════════════════════════════════════════════════════════════
#  AnalysisPipeline — Orquestador principal
# ══════════════════════════════════════════════════════════════
class AnalysisPipeline:
    """
    Orquestador del pipeline de análisis.

    Principio DIP: recibe INTERFACES, no clases concretas.
    Esto permite cambiar cualquier componente (loader, cleaner,
    analyzers, reporter) sin modificar este código.

    Parámetros
    ----------
    loader     : IDataLoader   — Responsable de cargar datos
    cleaner    : IDataCleaner  — Responsable de limpiar datos
    analyzers  : list[IAnalyzer]  — Lista de análisis a ejecutar
    visualizers: list[IVisualizer]— Lista de visualizaciones a generar
    reporter   : IReporter     — Responsable de reportar resultados
    """

    def __init__(
        self,
        loader: IDataLoader,
        cleaner: IDataCleaner,
        analyzers: list,
        visualizers: list,
        reporter: IReporter,
        generate_plots: bool = True
    ):
        self.loader         = loader
        self.cleaner        = cleaner
        self.analyzers      = analyzers
        self.visualizers    = visualizers
        self.reporter       = reporter
        self.generate_plots = generate_plots

    def run(self) -> dict:
        """
        Ejecuta el pipeline completo:
        1. Cargar → 2. Validar → 3. Limpiar → 4. Analizar → 5. Visualizar → 6. Reportar
        """
        logger.info("=" * 55)
        logger.info("🚀 Iniciando Customer Support Analysis Pipeline")
        logger.info("=" * 55)

        # ── Paso 1: Cargar datos ──────────────────────────────
        logger.info("📥 Paso 1/5: Cargando datos...")
        df_raw = self.loader.load()
        self.loader.validate(df_raw)

        # ── Paso 2: Limpiar datos ─────────────────────────────
        logger.info("🧹 Paso 2/5: Limpiando datos...")
        df_clean = self.cleaner.clean(df_raw)

        # ── Paso 3: Ejecutar análisis ─────────────────────────
        logger.info(f"📊 Paso 3/5: Ejecutando {len(self.analyzers)} análisis...")
        all_results = {}
        for analyzer in self.analyzers:
            logger.info(f"   → {analyzer.get_name()}")
            all_results[analyzer.get_name()] = analyzer.analyze(df_clean)

        # ── Paso 4: Visualizaciones ───────────────────────────
        if self.generate_plots:
            logger.info(f"🎨 Paso 4/5: Generando {len(self.visualizers)} visualizaciones...")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            for viz in self.visualizers:
                logger.info(f"   → {viz.get_title()}")
                viz.plot(df_clean, output_dir=str(OUTPUT_DIR))
        else:
            logger.info("⏭️  Paso 4/5: Visualizaciones omitidas (--no-plots)")

        # ── Paso 5: Reporte ───────────────────────────────────
        logger.info("📋 Paso 5/5: Generando reporte...")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.reporter.report(
            all_results,
            output_path=str(REPORTS_DIR / 'analysis_results.json')
        )

        logger.info("✅ Pipeline completado exitosamente.")
        return all_results


# ══════════════════════════════════════════════════════════════
#  Punto de entrada
# ══════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description='Customer Support Analysis Pipeline (SOLID)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                  # Análisis completo
  python main.py --sample         # Solo 5,000 filas (rápido)
  python main.py --no-plots       # Sin gráficos
        """
    )
    parser.add_argument('--sample', action='store_true',
                        help='Usar solo 5,000 filas para prueba rápida')
    parser.add_argument('--no-plots', action='store_true',
                        help='No generar visualizaciones')
    return parser.parse_args()


def main():
    args = parse_args()
    nrows = 5000 if args.sample else None

    # ─── Composición de dependencias (Dependency Injection) ───
    # DIP: los módulos de alto nivel dependen de abstracciones.
    # El "cableado" ocurre aquí, no dentro de las clases.

    loader = CSVDataLoader(
        file_path=str(DATA_FILE),
        nrows=nrows
    )

    cleaner = CustomerSupportCleaner()

    analyzers = [
        SentimentAnalyzer(),
        OutcomeAnalyzer(),
        VolumeAnalyzer(),
        UrgencyAnalyzer(),
        IntentAnalyzer(),
    ]

    visualizers = [
        SentimentVisualizer(),
        OutcomeVisualizer(),
        VolumeVisualizer(),
        IntentVisualizer(),
    ]

    reporter = ConsoleReporter()

    # ─── Ejecutar pipeline ────────────────────────────────────
    pipeline = AnalysisPipeline(
        loader=loader,
        cleaner=cleaner,
        analyzers=analyzers,
        visualizers=visualizers,
        reporter=reporter,
        generate_plots=not args.no_plots
    )

    results = pipeline.run()
    return results


if __name__ == "__main__":
    main()
