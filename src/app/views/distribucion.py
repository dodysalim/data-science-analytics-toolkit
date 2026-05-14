import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from .components import fig_bar, fig_pie

def render_distribucion(df_raw, df):
    st.title("📊 Distribucion de Tickets")

    col1, col2 = st.columns(2)
    with col1:
        by_ch = df["channel"].value_counts()
        st.pyplot(fig_pie(by_ch, "Por Canal"))
        plt.close()
    with col2:
        by_lang = df["language"].value_counts()
        st.pyplot(fig_pie(by_lang, "Por Idioma"))
        plt.close()

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        by_issue = df["issue_type"].value_counts().head(10)
        st.pyplot(fig_bar(by_issue, "Top 10 Tipos de Incidencia", h=True))
        plt.close()
    with col4:
        by_intent = df["primary_intent"].value_counts().head(10)
        st.pyplot(fig_bar(by_intent, "Top 10 Intenciones Primarias", h=True, color="#7c3aed"))
        plt.close()

    st.divider()
    st.subheader("Tickets por hora del dia")
    by_hora = df.groupby("hora")["conv_id"].count()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.fill_between(by_hora.index, by_hora.values, alpha=0.3, color="#2563eb")
    ax.plot(by_hora.index, by_hora.values, color="#2563eb", lw=2.5, marker="o", markersize=4)
    ax.set_xlabel("Hora del dia"); ax.set_ylabel("N conversaciones")
    ax.set_title("Distribucion horaria de tickets", fontweight="bold")
    ax.set_xticks(range(0,24))
    sns.despine(); fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.subheader("Filtro por industria")
    industria_sel = st.selectbox("Selecciona industria", ["Todas"] + sorted(df["industry"].unique()))
    df_f = df if industria_sel == "Todas" else df[df["industry"] == industria_sel]
    st.dataframe(df_f["product"].value_counts().reset_index().rename(
        columns={"index":"Producto","product":"Tickets"}), use_container_width=True)
