import streamlit as st
import matplotlib.pyplot as plt
from .components import fig_bar

def render_resumen(df_raw, df):
    st.title("🎧 Customer Support Analytics")
    st.subheader("Autor: **Dody Duenas** | Datos reales de soporte al cliente")
    st.divider()

    n_conv = df["conv_id"].nunique()
    n_neg  = (df["overall_sentiment"] == "negative").sum()
    n_res  = (df["outcome"] == "Resolved").sum()
    n_esc  = (df["outcome"] == "Escalated").sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Conversaciones", f"{n_conv:,}")
    c2.metric("Mensajes totales", f"{len(df_raw):,}")
    c3.metric("Sentimiento negativo", f"{n_neg/len(df):.1%}", delta="-")
    c4.metric("Tasa de resolucion", f"{n_res/len(df):.1%}", delta="+")
    c5.metric("Tasa escalada", f"{n_esc/len(df):.1%}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Volumen por Industria")
        by_ind = df["industry"].value_counts()
        st.pyplot(fig_bar(by_ind, "Conversaciones por Industria"))
    with col2:
        st.subheader("Distribucion de Sentimientos")
        by_sent = df["overall_sentiment"].value_counts()
        colores_sent = {"negative":"#ef4444","neutral":"#f59e0b","positive":"#22c55e"}
        fig, ax = plt.subplots(figsize=(4.5,4.5))
        ax.pie(by_sent.values, labels=by_sent.index, autopct="%1.1f%%",
               colors=[colores_sent.get(s,"#2563eb") for s in by_sent.index],
               wedgeprops={"width":0.55,"edgecolor":"white"})
        ax.set_title("Sentimientos", fontweight="bold")
        st.pyplot(fig); plt.close()

    st.subheader("Volumen mensual de tickets")
    vol_mes = df.groupby("mes")["conv_id"].count()
    st.line_chart(vol_mes)

    st.divider()
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(15)[["conv_id","timestamp","industry","product",
                               "issue_type","channel","overall_sentiment",
                               "overall_urgency","outcome","primary_intent"]],
                 use_container_width=True)
