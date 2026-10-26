# 13 — Kit de Pruebas A/B y Análisis de Experimentos

Framework estadístico completo para diseñar, ejecutar e informar experimentos controlados — enfoques frecuentista y bayesiano.

## Autor

**Dody Dueñas**

## Clases

| Clase | Descripción |
|---|---|
| `SampleSizeCalculator` | Análisis de potencia para proporciones y medias, cálculo del efecto mínimo detectable (EMD) |
| `FrequentistABTester` | Prueba Z, t de Welch, Mann-Whitney U, chi-cuadrado |
| `BayesianABTester` | Modelo Beta-Binomial conjugado, P(tratamiento > control), incremento esperado |
| `ExperimentReporter` | Generación de informes de experimentos A/B en Markdown |

## Flujo de Trabajo

```python
from ab_testing import SampleSizeCalculator, FrequentistABTester, BayesianABTester, ExperimentReporter

# 1. Planificar el tamaño de muestra
calc = SampleSizeCalculator()
plan = calc.for_proportions(baseline_rate=0.05, min_detectable_effect=0.20)
print(f"Se necesitan {plan['n_per_group']:,} usuarios por grupo")

# 2. Ejecutar prueba frecuentista
tester = FrequentistABTester()
resultado = tester.proportion_test(
    control_conversions=500, control_n=10000,
    treatment_conversions=580, treatment_n=10000,
)
print(resultado["recomendacion"])

# 3. Análisis bayesiano
bay = BayesianABTester()
resultado_bay = bay.analyze(500, 10000, 580, 10000)
print(f"P(tratamiento gana): {resultado_bay['prob_tratamiento_gana']:.1%}")

# 4. Generar informe
reporter = ExperimentReporter()
md = reporter.generate("Prueba Color Botón CTA", resultado, resultado_bay, plan)
```

## Dependencias

```
scipy>=1.10.0
numpy>=1.23.0
pandas>=1.5.0
```
