"""
📋 Reporter — Implementación de IReporter
==========================================
Genera un reporte de consola profesional con todos los resultados.

Principio de Responsabilidad Única (SRP):
  Solo se encarga de formatear y mostrar resultados.

Principio de Inversión de Dependencias (DIP):
  Depende de la abstracción IReporter, no de implementaciones concretas.

Autor: Dody Dueñas
"""

import json
from pathlib import Path
from typing import Optional
from interfaces import IReporter


class ConsoleReporter(IReporter):
    """
    Genera un reporte detallado en consola con formato legible.
    Opcionalmente guarda un JSON con todos los resultados.
    """

    def report(self, results: dict, output_path: Optional[str] = None) -> None:
        """
        Imprime el reporte en consola y opcionalmente guarda JSON.

        Parámetros
        ----------
        results : dict
            Diccionario {nombre_analisis: resultados_dict}
        output_path : str, opcional
            Ruta donde guardar el JSON. Si None, solo imprime en consola.
        """
        print("\n" + "═" * 70)
        print("  📊  REPORTE DE ANÁLISIS — CUSTOMER SUPPORT DATASET")
        print("═" * 70)

        for analysis_name, data in results.items():
            print(f"\n{'─' * 70}")
            print(f"  🔹  {analysis_name.upper()}")
            print(f"{'─' * 70}")
            self._print_dict(data, indent=2)

        print("\n" + "═" * 70)
        print("  ✅  Análisis completado exitosamente.")
        print("═" * 70)

        if output_path:
            self._save_json(results, output_path)

    def _print_dict(self, data: dict, indent: int = 0) -> None:
        """Imprime un diccionario con formato jerárquico."""
        prefix = " " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}📌 {key}:")
                self._print_dict(value, indent + 4)
            elif isinstance(value, list):
                print(f"{prefix}📌 {key}: {value}")
            elif isinstance(value, float):
                print(f"{prefix}▸ {key}: {value:.2f}")
            else:
                print(f"{prefix}▸ {key}: {value}")

    def _save_json(self, results: dict, path: str) -> None:
        """Guarda los resultados en un archivo JSON."""
        # Convertir claves tupla a string para JSON serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(i) for i in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj

        clean_results = make_serializable(results)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 Reporte JSON guardado en: {path}")


class JSONReporter(IReporter):
    """
    Alternativa: guarda los resultados SOLO en JSON sin imprimir en consola.
    Demuestra OCP: nueva forma de reportar sin cambiar ConsoleReporter.
    """

    def report(self, results: dict, output_path: Optional[str] = None) -> None:
        if not output_path:
            raise ValueError("JSONReporter requiere un output_path.")

        def make_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(i) for i in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj

        clean = make_serializable(results)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        print(f"  💾 JSON guardado: {output_path}")
