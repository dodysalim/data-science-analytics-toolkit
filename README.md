<p align="center"><img src="https://dodysalim.github.io/assets/project-covers/toolkit.webp" alt="Customer Support Analytics" width="100%"></p>

<h1 align="center">Customer Support Analytics</h1>

<p align="center"><b>Métricas, calidad de datos y análisis de texto sobre conversaciones de soporte.</b></p>

<p align="center">
<img src="https://img.shields.io/badge/Proyecto_personal-75E6CE?style=for-the-badge&labelColor=0b1222" alt="Proyecto personal">
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas"> <img src="https://img.shields.io/badge/NLP-8F7BFF?style=flat-square&logoColor=white" alt="NLP"> <img src="https://img.shields.io/badge/SQL-4169E1?style=flat-square&logo=postgresql&logoColor=white" alt="SQL"> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"> <img src="https://img.shields.io/badge/Power_BI-F2C811?style=flat-square&logoColor=black" alt="Power BI">
</p>

<p align="center"><a href="https://dodysalim.github.io/"><b>🌐 Portafolio</b></a> · <a href="docs/PORTFOLIO_CASE.md"><b>📄 Caso de estudio</b></a> · <a href="docs/VALIDATION.md"><b>✅ Verificación</b></a></p>

---

## 📌 Resumen en 30 segundos

| | |
| --- | --- |
| **🎯 Problema** | Descubrir patrones de canal, sentimiento, urgencia y resultado en conversaciones de soporte. |
| **🛠️ Solución** | Módulos reutilizables de limpieza, estadística, características y evaluación, más análisis SQL. |
| **📈 Entregables** | Dashboard Streamlit de 7 vistas (incluye NLP y A/B) y Power BI de 7 páginas. |
| **🧪 Calidad** | 7 vistas probadas sin errores con 50.000 filas reales. |
| **👤 Autor** | Dody Dueñas Remache. |

## ❓ La pregunta de negocio

¿Qué patrones de canal, sentimiento y resultado aparecen en las conversaciones de soporte?

## ✨ Qué incluye

- Siete vistas: resumen, distribución, sentimientos, urgencia, NLP, A/B y calidad.
- Módulos reutilizables de limpieza, estadísticas, características y evaluación.
- Análisis SQL, notebooks y Power BI de siete páginas.

## 🚀 Cómo ejecutarlo

Usa Python 3.11 o 3.12 en un entorno independiente. Desde la raíz:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
# Dataset versionado con Git LFS:
git lfs pull
```

Después de configurar los datos:

```bash
python -m streamlit run src/app/dashboard.py
```

## 🗂️ Datos y configuración

El CSV grande requiere Git LFS. También puedes definir CUSTOMER_SUPPORT_DATA con la ruta de un CSV disponible. El dashboard carga primeras filas, no una muestra aleatoria.

## 📊 Power BI · PC y móvil

[Archivos e instrucciones](powerbi/README.md). Descarga el repositorio completo y abre `powerbi/Abrir-PowerBI.bat` en Windows; después pulsa **Actualizar**. Incluye A4 horizontal a tamaño real (100 %) y diseño móvil vertical. El archivo `.pbip` necesita sus carpetas Report, SemanticModel y data.

## 🧭 Estructura del repositorio

| Ruta | Qué contiene |
| --- | --- |
| [src/app/](src/app/) | Dashboard y vistas |
| [src/modules/](src/modules/) | Motores reutilizables |
| [notebooks/](notebooks/) | EDA, SQL y soporte |
| [data/](data/) | Dataset versionado mediante LFS |

## ✅ Calidad y alcance

Siete vistas probadas sin excepciones con una muestra de 50.000 filas originales. No existe cobertura global de pruebas para todos los módulos.

Las conversaciones se deduplican por conv_id para los KPIs. La comparación A/B es observacional y no demuestra causalidad. El NLP usa un subconjunto; las tablas precalculadas de Power BI conservan ese alcance.

## 👥 Autoría

Proyecto personal. Dody Salim Dueñas Remache.

[Documentación anterior](docs/ORIGINAL_README.md), conservada como referencia histórica.

---

<p align="center">
<b>Dody Dueñas Remache</b> · Analista de Datos / Data Scientist<br>
<a href="https://dodysalim.github.io/">Portafolio</a> · <a href="https://github.com/dodysalim">GitHub</a> · <a href="https://www.linkedin.com/in/dody-duenas/">LinkedIn</a> · <a href="mailto:dodydurema67@gmail.com">dodydurema67@gmail.com</a>
</p>
