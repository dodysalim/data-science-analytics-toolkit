import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def render_sentimientos(df_raw, df):
    st.title("😊 Analisis de Sentimientos")

    sent_cols = {"negative":"#ef4444","neutral":"#f59e0b","positive":"#22c55e"}

    st.subheader("Sentimiento por Canal")
    pivot_ch = df.groupby(["channel","overall_sentiment"]).size().unstack(fill_value=0)
    pivot_ch_pct = pivot_ch.div(pivot_ch.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8,4))
    pivot_ch_pct[["negative","neutral","positive"]].plot(
        kind="bar", ax=ax, stacked=True,
        color=["#ef4444","#f59e0b","#22c55e"],
        edgecolor="white")
    ax.set_title("Distribucion de sentimiento por canal", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Proporcion")
    ax.legend(title="Sentimiento", bbox_to_anchor=(1,1))
    plt.xticks(rotation=30, ha="right")
    sns.despine(); fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentimiento por Industria")
        pivot_ind = df.groupby(["industry","overall_sentiment"]).size().unstack(fill_value=0)
        pivot_ind_pct = pivot_ind.div(pivot_ind.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(6,4))
        pivot_ind_pct[["negative","neutral","positive"]].plot(
            kind="barh", ax=ax, stacked=True,
            color=["#ef4444","#f59e0b","#22c55e"], edgecolor="white")
        ax.set_title("Sentimiento por industria", fontweight="bold")
        ax.legend(title="Sentimiento", bbox_to_anchor=(1,1))
        sns.despine(); fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Sentimiento por Urgencia")
        orden_urg = ["low","medium","high","critical"]
        pivot_urg = df.groupby(["overall_urgency","overall_sentiment"]).size().unstack(fill_value=0)
        pivot_urg = pivot_urg.reindex([u for u in orden_urg if u in pivot_urg.index])
        pivot_urg_pct = pivot_urg.div(pivot_urg.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(5,4))
        pivot_urg_pct[["negative","neutral","positive"]].plot(
            kind="bar", ax=ax, stacked=True,
            color=["#ef4444","#f59e0b","#22c55e"], edgecolor="white")
        ax.set_title("Sentimiento vs Urgencia", fontweight="bold")
        plt.xticks(rotation=20)
        sns.despine(); fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Evolucion mensual del sentimiento negativo")
    neg_mes = df[df["overall_sentiment"]=="negative"].groupby("mes").size()
    tot_mes = df.groupby("mes").size()
    pct_neg = (neg_mes / tot_mes * 100).fillna(0)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(pct_neg.index, pct_neg.values, color="#ef4444", lw=2.5, marker="o", markersize=4)
    ax.fill_between(pct_neg.index, pct_neg.values, alpha=0.15, color="#ef4444")
    ax.set_ylabel("% Negativo"); ax.set_title("Tasa de sentimiento negativo mensual", fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=7)
    sns.despine(); fig.tight_layout()
    st.pyplot(fig); plt.close()
