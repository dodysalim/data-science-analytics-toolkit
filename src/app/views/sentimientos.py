import streamlit as st

from .analytics import monthly_rate, share_table
from .components import fig_area, fig_stacked_share, show
from .theme import CORAL, SENTIMENT_ORDER, URGENCY_ORDER


def render_sentimientos(df_raw, df):
    st.title("😊 Analisis de Sentimientos")

    st.subheader("Sentimiento por Canal")
    show(fig_stacked_share(share_table(df, "channel", "overall_sentiment", columns=SENTIMENT_ORDER),
                           "Distribucion de sentimiento por canal"))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentimiento por Industria")
        show(fig_stacked_share(share_table(df, "industry", "overall_sentiment", columns=SENTIMENT_ORDER),
                               "Sentimiento por industria", horizontal=True))
    with col2:
        st.subheader("Sentimiento por Urgencia")
        show(fig_stacked_share(share_table(df, "overall_urgency", "overall_sentiment", order=URGENCY_ORDER,
                                           columns=SENTIMENT_ORDER), "Sentimiento vs Urgencia"))

    st.divider()
    st.subheader("Evolucion mensual del sentimiento negativo")
    show(fig_area(monthly_rate(df, "overall_sentiment", "negative"), "Tasa de sentimiento negativo mensual",
                  color=CORAL, y_title="% Negativo"))
