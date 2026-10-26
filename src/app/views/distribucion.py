import streamlit as st

from .components import fig_area, fig_bar, fig_pie, show
from .theme import VIOLET


def render_distribucion(df_raw, df):
    st.title("📊 Distribucion de Tickets")

    col1, col2 = st.columns(2)
    with col1:
        show(fig_pie(df["channel"].value_counts(), "Por Canal"))
    with col2:
        show(fig_pie(df["language"].value_counts(), "Por Idioma"))

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        show(fig_bar(df["issue_type"].value_counts().head(10), "Top 10 Tipos de Incidencia", h=True))
    with col4:
        show(fig_bar(df["primary_intent"].value_counts().head(10), "Top 10 Intenciones Primarias", h=True, color=VIOLET))

    st.divider()
    st.subheader("Tickets por hora del dia")
    fig = fig_area(df.groupby("hora")["conv_id"].count(), "Distribucion horaria de tickets",
                   x_title="Hora del dia", y_title="N conversaciones")
    fig.update_xaxes(dtick=1)
    show(fig)

    st.subheader("Filtro por industria")
    industria = st.selectbox("Selecciona industria", ["Todas"] + sorted(df["industry"].unique()))
    subset = df if industria == "Todas" else df[df["industry"] == industria]
    st.dataframe(subset["product"].value_counts().rename_axis("Producto").reset_index(name="Tickets"),
                 use_container_width=True)
