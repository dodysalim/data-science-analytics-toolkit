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
DATA = os.environ.get("CUSTOMER_SUPPORT_DATA", os.path.join(REPO_DIR, "data", "customer_support_data.csv"))

for mod in ["12_data_profiling","11_nlp_toolkit","13_ab_testing"]:
    sys.path.insert(0, os.path.join(SRC_DIR, "modules", mod))

from views.resumen import render_resumen
from views.distribucion import render_distribucion
from views.sentimientos import render_sentimientos
from views.resultados import render_resultados
from views.nlp import render_nlp
from views.ab_testing import render_ab_testing
from views.calidad import render_calidad
from views.theme import CSS

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Support Analytics — Dody Duenas",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎧 Customer Support Analytics")
# Registro de páginas: agregar una vista = agregar una entrada
PAGES = {
    "🏠 Resumen Ejecutivo": render_resumen,
    "📊 Distribucion de Tickets": render_distribucion,
    "😊 Analisis de Sentimientos": render_sentimientos,
    "⏱️ Urgencia & Resultados": render_resultados,
    "💬 Analisis de Texto (NLP)": render_nlp,
    "🧪 Prueba A/B de Canales": render_ab_testing,
    "🔍 Perfil de Calidad": render_calidad,
}
pagina = st.sidebar.radio("Seccion", list(PAGES))
st.sidebar.divider()

n_filas = st.sidebar.slider("Filas a cargar (miles)", 10, 200, 50) * 1000
st.sidebar.caption("**Autor:** Dody Duenas")
st.sidebar.caption(f"Fuente: customer_support_data.csv")

# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos reales...")
def cargar_datos(n):
    with open(DATA, encoding="utf-8") as source:
        if source.readline().startswith("version https://git-lfs.github.com/spec/"):
            raise ValueError("Falta descargar el dataset: ejecuta git lfs pull en la raíz del repositorio o configura CUSTOMER_SUPPORT_DATA con un CSV disponible.")
    df = pd.read_csv(DATA, nrows=n, parse_dates=["timestamp"])
    conv = df[df["role"] == "customer"].drop_duplicates("conv_id").copy()
    conv["fecha"] = pd.to_datetime(conv["timestamp"]).dt.date
    conv["hora"]  = pd.to_datetime(conv["timestamp"]).dt.hour
    conv["mes"]   = pd.to_datetime(conv["timestamp"]).dt.to_period("M").astype(str)
    return df, conv

try:
    df_raw, df = cargar_datos(n_filas)
except (OSError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

# ── Router ────────────────────────────────────────────────────────────────────
PAGES[pagina](df_raw, df)
