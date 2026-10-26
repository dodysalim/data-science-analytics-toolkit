import streamlit as st

from .analytics import monthly_rate
from .components import fig_area, fig_bar, fig_pie, show
from .theme import hero

PREVIEW_COLUMNS = ["conv_id", "timestamp", "industry", "product", "issue_type", "channel",
                   "overall_sentiment", "overall_urgency", "outcome", "primary_intent"]


def render_resumen(df_raw, df):
    st.markdown(hero("🎧 Customer Support", "Analytics",
                     "Autor: <b>Dody Duenas</b> · Datos reales de soporte al cliente"), unsafe_allow_html=True)

    n = len(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Conversaciones", f"{df['conv_id'].nunique():,}")
    c2.metric("Mensajes totales", f"{len(df_raw):,}")
    c3.metric("Sentimiento negativo", f"{(df['overall_sentiment'] == 'negative').sum() / n:.1%}")
    c4.metric("Tasa de resolucion", f"{(df['outcome'] == 'Resolved').sum() / n:.1%}")
    c5.metric("Tasa escalada", f"{(df['outcome'] == 'Escalated').sum() / n:.1%}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Volumen por Industria")
        show(fig_bar(df["industry"].value_counts(), "Conversaciones por Industria"))
    with col2:
        st.subheader("Distribucion de Sentimientos")
        show(fig_pie(df["overall_sentiment"].value_counts(), "Sentimientos"))

    st.subheader("Volumen mensual de tickets")
    show(fig_area(df.groupby("mes")["conv_id"].count(), "Tickets por mes", y_title="Conversaciones"))

    st.divider()
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(15)[PREVIEW_COLUMNS], use_container_width=True)
