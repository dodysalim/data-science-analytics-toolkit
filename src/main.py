"""
# ============================================================================
#   DATA SCIENCE ANALYTICS TOOLKIT - ORQUESTADOR SOLID
#   Autor: Dody Duenas
# ============================================================================
"""
import sys
import os
import pandas as pd
import numpy as np

# Configuración de encoding para Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Agregamos la raiz del proyecto al path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.patterns.observer import ConsoleLogger
from src.core.data_loader import DataLoader
from src.core.pipeline_builder import PipelineBuilder

def main():
    DATA_PATH = os.path.join(ROOT_DIR, "data", "customer_support_data.csv")

    # Inicializar Observer (Logger)
    logger = ConsoleLogger()
    logger.update("Main", "INFO", f"Cargando datos desde {DATA_PATH}...")

    # Carga y Preparacion de Datos (Modularizada)
    df_base = DataLoader.load_and_prepare(DATA_PATH, nrows=5000)

    # Construccion del Pipeline Facade (Modularizada)
    facade = PipelineBuilder.build(logger)

    # Ejecutar Pipeline Maestra
    logger.update("Main", "START", "Ejecutando arquitectura SOLID modularizada")
    final_context = facade.execute_pipeline(df_base)
    logger.update("Main", "END", "Ejecución completada. ¡Módulos ejecutados exitosamente!")

if __name__ == "__main__":
    main()
