from .interfaces import IObserver

class ConsoleLogger(IObserver):
    """
    Implementación concreta de IObserver para registrar en consola.
    """
    def __init__(self):
        self.sep = "=" * 75

    def update(self, step_name: str, status: str, message: str) -> None:
        if status == "START":
            print(f"\n{self.sep}")
            print(f" 🚀 INICIANDO MÓDULO: {step_name}")
            print(f"{self.sep}")
        elif status == "SUCCESS":
            print(f"  [✅ OK] {message}")
        elif status == "INFO":
            print(f"  [ℹ️ INFO] {message}")
        elif status == "ERROR":
            print(f"  [❌ ERROR] {step_name}: {message}")
        elif status == "END":
            print(f" 🏁 FINALIZADO: {step_name}\n")
