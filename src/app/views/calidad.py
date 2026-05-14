import streamlit as st
from data_profiling import DatasetProfiler

def render_calidad(df_raw, df):
    st.title("🔍 Perfil de Calidad del Dataset")

    cols_perfil = ["industry","product","issue_type","language","channel",
                   "overall_sentiment","overall_urgency","outcome","primary_intent"]
    df_perfil = df[cols_perfil].copy()

    with st.spinner("Perfilando columnas..."):
        perfilador = DatasetProfiler()
        reporte = perfilador.profile(df_perfil)

    pc = reporte["puntuacion_calidad"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Completitud",  f"{pc['completitud']}%")
    c2.metric("Unicidad",     f"{pc['unicidad']}%")
    c3.metric("Consistencia", f"{pc['consistencia']}%")
    c4.metric("Calificacion", pc["calificacion"], f"{pc['total']}/100")

    st.divider()
    st.subheader("Tabla de perfil por columna")
    tabla = perfilador.summary_table(reporte)
    st.dataframe(tabla.round(3), use_container_width=True)

    st.subheader("Valores nulos por columna")
    nulos = df[cols_perfil].isna().sum().reset_index()
    nulos.columns = ["Columna","Nulos"]
    nulos["Pct"] = (nulos["Nulos"]/len(df)*100).round(2)
    st.dataframe(nulos, use_container_width=True)
