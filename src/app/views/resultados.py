import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from .components import fig_pie, PALETTE

def render_resultados(df_raw, df):
    st.title("⏱️ Urgencia & Resultados")

    c1,c2,c3,c4 = st.columns(4)
    for metric, col in zip(["critical","high","medium","low"],
                            [c1,c2,c3,c4]):
        n = (df["overall_urgency"]==metric).sum()
        col.metric(metric.capitalize(), f"{n:,}", f"{n/len(df):.1%}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Outcomes (resultados)")
        by_out = df["outcome"].value_counts()
        cols_out = ["#22c55e","#94a3b8","#f59e0b","#ef4444","#7c3aed"]
        st.pyplot(fig_pie(by_out, "Distribucion de Resultados"))
        plt.close()

    with col2:
        st.subheader("Resultado por Urgencia")
        pivot_out = df.groupby(["overall_urgency","outcome"]).size().unstack(fill_value=0)
        pivot_out = pivot_out.reindex([u for u in ["low","medium","high","critical"] if u in pivot_out.index])
        pivot_out_pct = pivot_out.div(pivot_out.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(6,4))
        pivot_out_pct.plot(kind="bar", ax=ax, stacked=True,
                           color=PALETTE[:len(pivot_out_pct.columns)], edgecolor="white")
        ax.set_title("Resultado segun urgencia", fontweight="bold")
        ax.set_xlabel(""); plt.xticks(rotation=20)
        ax.legend(bbox_to_anchor=(1,1), fontsize=7)
        sns.despine(); fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Tasa de resolucion por industria")
    res_ind = df.groupby("industry").apply(
        lambda x: (x["outcome"]=="Resolved").sum() / len(x) * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,3))
    bars = ax.bar(res_ind.index, res_ind.values, color="#2563eb", edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set_ylabel("% Resuelto"); ax.set_ylim(0, 100)
    ax.set_title("Tasa de resolucion por industria", fontweight="bold")
    sns.despine(); fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.subheader("Heatmap: Issue Type vs Outcome")
    pivot_heat = df.groupby(["issue_type","outcome"]).size().unstack(fill_value=0)
    pivot_heat_pct = pivot_heat.div(pivot_heat.sum(axis=1), axis=0).round(2)
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot_heat_pct)//2)))
    sns.heatmap(pivot_heat_pct, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax, cbar_kws={"label":"Proporcion"})
    ax.set_title("Proporcion de outcomes por tipo de incidencia", fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig); plt.close()
