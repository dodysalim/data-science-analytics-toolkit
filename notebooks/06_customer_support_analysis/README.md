# 🎧 Customer Support Analysis — SOLID Architecture

Análisis completo de un dataset real de **Customer Support** con **976,271 conversaciones** de 8 industrias distintas. El proyecto aplica los **5 principios SOLID** en su arquitectura.

## 📊 Dataset

El archivo `data/customer_support_sample.csv` contiene **20,000 conversaciones estratificadas** (muestra representativa del dataset original de 976K filas).

| Campo | Descripción |
|-------|-------------|
| `conv_id` | ID único de conversación |
| `turn_index` | Número de turno en la conversación |
| `role` | `customer` o `agent` |
| `text` | Texto del mensaje |
| `timestamp` | Fecha y hora del mensaje |
| `industry` | Industria (SaaS, Travel, Telecom, FoodTech, EdTech, E-commerce, Fintech, HealthTech) |
| `overall_sentiment` | `positive`, `neutral`, `negative` |
| `overall_urgency` | `low`, `medium`, `high`, `critical` |
| `outcome` | Resultado: `Resolved`, `Escalated`, `Pending Customer`, etc. |
| `primary_intent` | Intención del cliente (`refund_status`, `cancel_order`, etc.) |

## 🏗️ Arquitectura SOLID

```
06_customer_support_analysis/
├── main.py                    ← Orquestador (DIP: inyección de dependencias)
├── src/
│   ├── interfaces.py          ← Contratos abstractos (ISP, DIP)
│   ├── data_loaders.py        ← Carga de datos CSV/Parquet (SRP, OCP)
│   ├── data_cleaner.py        ← Limpieza de datos (SRP, LSP)
│   ├── analyzers.py           ← 5 análisis independientes (SRP, OCP, LSP)
│   ├── visualizers.py         ← 4 visualizadores (SRP, OCP)
│   └── reporters.py           ← Console/JSON reporters (SRP, OCP, DIP)
├── data/
│   └── customer_support_sample.csv
├── output/                    ← Gráficos generados automáticamente
└── reports/                   ← Reporte JSON de resultados
```

## 🔷 Principios SOLID Aplicados

| Principio | Dónde se aplica | Ejemplo concreto |
|-----------|----------------|------------------|
| **S** — Single Responsibility | Todas las clases | `CustomerSupportCleaner` solo limpia. `ConsoleReporter` solo reporta. |
| **O** — Open/Closed | `analyzers.py`, `visualizers.py` | Agregar un nuevo análisis = nueva clase. Nada se modifica. |
| **L** — Liskov Substitution | `IAnalyzer`, `IVisualizer` | Cualquier `IAnalyzer` puede reemplazarse por otro sin romper el pipeline. |
| **I** — Interface Segregation | `interfaces.py` | 5 interfaces pequeñas y específicas. No hay métodos innecesarios. |
| **D** — Dependency Inversion | `main.py` | El orquestador depende de abstracciones, no de `CSVDataLoader` concreto. |

## 📈 Análisis Incluidos

1. **Análisis de Sentimiento** — Distribución por industria, canal y producto
2. **Análisis de Outcomes** — Tasa de resolución, escalación por urgencia
3. **Análisis de Volumen** — Hora pico, día pico, duración de conversaciones
4. **Análisis de Urgencia** — Urgencia crítica por industria, relación con sentimiento
5. **Análisis de Intenciones** — Top intenciones, correlación con escalación

## 🚀 Ejecución

```bash
# Instalar dependencias
pip install -r requirements.txt

# Análisis completo (con gráficos)
python main.py

# Modo rápido (5,000 filas, sin gráficos)
python main.py --sample --no-plots
```

## 🎨 Visualizaciones Generadas

Al ejecutar el pipeline se generan automáticamente en `output/`:

- `sentiment_analysis.png` — Distribución de sentimiento
- `outcome_analysis.png` — Resultados de conversaciones
- `volume_analysis.png` — Temporalidad y volumen
- `intent_analysis.png` — Intenciones del cliente

---
> **Autor:** Dody Dueñas | Data Analyst & Data Scientist
> **Dataset:** 976,271 conversaciones | 8 industrias | 16 variables
