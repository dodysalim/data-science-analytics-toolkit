"""
Dashboard — Customer Support Analytics
Autor: Dody Duenas
Datos: customer_support_data.csv (real)
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(APP_DIR)
REPO_DIR = os.path.dirname(SRC_DIR)
DATA = os.path.join(REPO_DIR, "data", "customer_support_data.csv")

for mod in ["12_data_profiling","11_nlp_toolkit","13_ab_testing"]:
    sys.path.insert(0, os.path.join(SRC_DIR, "modules", mod))

from views.resumen import render_resumen
from views.distribucion import render_distribucion
from views.sentimientos import render_sentimientos
from views.resultados import render_resultados
from views.nlp import render_nlp
from views.ab_testing import render_ab_testing
from views.calidad import render_calidad

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Support Analytics — Dody Duenas",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#f8fafc; color:#1e293b; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1e3a5f 0%,#2563eb 100%);
}
[data-testid="stSidebar"] * { color:#ffffff !important; }
[data-testid="metric-container"] {
    background:#ffffff; border:1px solid #e2e8f0;
    border-radius:12px; padding:16px;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
}
h1 { color:#1e3a5f !important; border-bottom:3px solid #2563eb; padding-bottom:8px; }
h2,h3 { color:#1e40af !important; }
hr  { border-color:#bfdbfe; }
.stButton>button {
    background:#2563eb; color:white; border-radius:8px;
    border:none; padding:8px 24px; font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎧 Customer Support Analytics")
pagina = st.sidebar.radio("Seccion", [
    "🏠 Resumen Ejecutivo",
    "📊 Distribucion de Tickets",
    "😊 Analisis de Sentimientos",
    "⏱️ Urgencia & Resultados",
    "💬 Analisis de Texto (NLP)",
    "🧪 Prueba A/B de Canales",
    "🔍 Perfil de Calidad",
])
st.sidebar.divider()

n_filas = st.sidebar.slider("Filas a cargar (miles)", 10, 200, 50) * 1000
st.sidebar.caption("**Autor:** Dody Duenas")
st.sidebar.caption(f"Fuente: customer_support_data.csv")

# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos reales...")
def cargar_datos(n):
    df = pd.read_csv(DATA, nrows=n, parse_dates=["timestamp"])
    conv = df[df["role"] == "customer"].drop_duplicates("conv_id")
    conv["fecha"] = pd.to_datetime(conv["timestamp"]).dt.date
    conv["hora"]  = pd.to_datetime(conv["timestamp"]).dt.hour
    conv["mes"]   = pd.to_datetime(conv["timestamp"]).dt.to_period("M").astype(str)
    return df, conv

df_raw, df = cargar_datos(n_filas)

# ── Router ────────────────────────────────────────────────────────────────────
if pagina == "🏠 Resumen Ejecutivo":
    render_resumen(df_raw, df)
elif pagina == "📊 Distribucion de Tickets":
    render_distribucion(df_raw, df)
elif pagina == "😊 Analisis de Sentimientos":
    render_sentimientos(df_raw, df)
elif pagina == "⏱️ Urgencia & Resultados":
    render_resultados(df_raw, df)
elif pagina == "💬 Analisis de Texto (NLP)":
    render_nlp(df_raw, df)
elif pagina == "🧪 Prueba A/B de Canales":
    render_ab_testing(df_raw, df)
elif pagina == "🔍 Perfil de Calidad":
    render_calidad(df_raw, df)
