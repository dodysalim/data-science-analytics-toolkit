<p align="center"><img src="https://dodysalim.github.io/assets/project-covers/toolkit.webp" alt="Customer Support Analytics" width="100%"></p>

<h1 align="center">Customer Support Analytics</h1>

<p align="center"><b>Metrics, data quality and text analysis over real customer-support conversations.</b></p>

<p align="center"><a href="https://data-science-analytics-toolkit-emhnwgam9hemykfa8qe38r.streamlit.app/"><b>Live demo ↗</b></a></p>

<p align="center"><img src="https://img.shields.io/badge/Personal_project-75E6CE?style=for-the-badge&labelColor=0b1222" alt="Personal project"></p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas"> <img src="https://img.shields.io/badge/NLP-8F7BFF?style=flat-square&logoColor=white" alt="NLP"> <img src="https://img.shields.io/badge/SQL-4169E1?style=flat-square&logo=postgresql&logoColor=white" alt="SQL"> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"> <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly"> <img src="https://img.shields.io/badge/Power_BI-F2C811?style=flat-square&logoColor=black" alt="Power BI">
</p>

<p align="center"><a href="https://dodysalim.github.io/"><b>🌐 Portfolio</b></a> · <a href="README.md"><b>🇪🇸 Versión en español</b></a></p>

---

## 📌 At a glance

| | |
| --- | --- |
| **🎯 Problem** | Find channel, sentiment, urgency and outcome patterns in support conversations. |
| **🛠️ Solution** | Reusable modules for cleaning, statistics, A/B testing, NLP and data profiling. |
| **📈 Deliverables** | Interactive 7-view Streamlit dashboard and a 7-page Power BI report. |
| **🧪 Quality** | 12 automated tests run in CI; views tested with 50,000 real rows. |

## 🏗️ Architecture

```mermaid
flowchart LR
    CSV[(customer_support_data.csv<br/>Git LFS)] --> LOAD[Carga con caché]
    LOAD --> MODS
    subgraph MODS[Módulos reutilizables]
        C[Limpieza] --- S[Estadística] --- AB[Pruebas A/B] --- N[NLP] --- P[Perfilado]
    end
    MODS --> AN[views/analytics<br/>cálculos puros]
    AN --> CH[views/components<br/>fábrica Plotly]
    CH --> UI[Dashboard · 7 vistas<br/>registro de páginas]
    MODS --> PBI[Power BI · 7 páginas]
```

## 🚀 How to run

```bash
pip install -r requirements.txt
git lfs pull   # large CSV
python -m streamlit run src/app/dashboard.py
python -m pytest tests -q
```

## ⚖️ Scope

The A/B comparison is observational and does not prove causality.

---

<p align="center">
<b>Dody Dueñas Remache</b> · Data Analyst / Data Scientist · Guayaquil, Ecuador (remote LATAM)<br>
<a href="https://dodysalim.github.io/">Portfolio</a> · <a href="https://github.com/dodysalim">GitHub</a> · <a href="https://www.linkedin.com/in/dody-duenas/">LinkedIn</a> · <a href="mailto:dodydurema67@gmail.com">dodydurema67@gmail.com</a>
</p>
