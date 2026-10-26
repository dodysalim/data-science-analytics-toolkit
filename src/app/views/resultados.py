import streamlit as st

from .analytics import rate_by, share_table
from .components import fig_bar, fig_heatmap, fig_pie, fig_stacked_share, show
from .theme import URGENCY_ORDER

OUTCOME_COLORS = {"Resolved": "#75E6CE", "Escalated": "#FF8A80", "Pending": "#F8D66D"}


def render_resultados(df_raw, df):
    st.title("⏱️ Urgencia & Resultados")

    for col, level in zip(st.columns(4), ["critical", "high", "medium", "low"]):
        n = (df["overall_urgency"] == level).sum()
        col.metric(level.capitalize(), f"{n:,}", f"{n / len(df):.1%}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Outcomes (resultados)")
        outcomes = df["outcome"].value_counts()
        show(fig_pie(outcomes, "Distribucion de Resultados",
                     colors=[OUTCOME_COLORS.get(o) or c for o, c in zip(outcomes.index, ["#88B7FF", "#C99BFF", "#5BC0EB"] * 3)]))
    with col2:
        st.subheader("Resultado por Urgencia")
        show(fig_stacked_share(share_table(df, "overall_urgency", "outcome", order=URGENCY_ORDER),
                               "Resultado segun urgencia", colors=OUTCOME_COLORS))

    st.divider()
    st.subheader("Tasa de resolucion por industria")
    fig = fig_bar(rate_by(df, "industry", "outcome", "Resolved"), "Tasa de resolucion por industria", text_fmt="{:.1f}%")
    fig.update_yaxes(range=[0, 100], title="% Resuelto")
    show(fig)

    st.subheader("Heatmap: Issue Type vs Outcome")
    show(fig_heatmap(share_table(df, "issue_type", "outcome").round(2), "Proporcion de outcomes por tipo de incidencia"))
