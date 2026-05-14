import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ab_testing import FrequentistABTester, BayesianABTester

def render_ab_testing(df_raw, df):
    st.title("🧪 Prueba A/B — Comparacion de Canales")

    st.info("Comparamos la **tasa de resolucion** entre dos canales de soporte usando los datos reales.")

    canales = sorted(df["channel"].unique())
    col1, col2 = st.columns(2)
    canal_a = col1.selectbox("Canal Control (A)", canales, index=0)
    canal_b = col2.selectbox("Canal Tratamiento (B)", canales,
                              index=min(1, len(canales)-1))

    df_a = df[df["channel"] == canal_a]
    df_b = df[df["channel"] == canal_b]

    conv_a = (df_a["outcome"] == "Resolved").sum()
    conv_b = (df_b["outcome"] == "Resolved").sum()
    n_a, n_b = len(df_a), len(df_b)

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"N {canal_a}", f"{n_a:,}")
    c2.metric(f"Resueltos {canal_a}", f"{conv_a:,}", f"{conv_a/n_a:.1%}")
    c3.metric(f"N {canal_b}", f"{n_b:,}")
    c4.metric(f"Resueltos {canal_b}", f"{conv_b:,}", f"{conv_b/n_b:.1%}")

    st.divider()
    tester = FrequentistABTester()
    res_z  = tester.proportion_test(int(conv_a), int(n_a), int(conv_b), int(n_b))

    st.subheader("Resultado Frecuentista (Prueba Z)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tasa Canal A",  f"{res_z['tasa_control']:.2%}")
    c2.metric("Tasa Canal B",  f"{res_z['tasa_tratamiento']:.2%}")
    c3.metric("Incremento",    f"{res_z['incremento_relativo_pct']:+.2f}%")
    c4.metric("p-valor",       f"{res_z['p_valor']:.5f}")

    if res_z["es_significativo"]:
        st.success(f"✅ **DIFERENCIA SIGNIFICATIVA** — {res_z['recomendacion']}")
    else:
        st.warning(f"⚠️ **Sin diferencia significativa** — {res_z['recomendacion']}")
    st.info(f"Z-stat: {res_z['estadistico_z']:.4f} | IC 95%: {res_z['intervalo_confianza_95pct']}")

    st.subheader("Analisis Bayesiano")
    bay = BayesianABTester()
    res_bay = bay.analyze(int(conv_a), int(n_a), int(conv_b), int(n_b))
    col1, col2 = st.columns(2)
    col1.metric("P(B > A)", f"{res_bay['prob_tratamiento_gana']:.1%}")
    col2.metric("Incremento esperado", f"{res_bay['media_incremento_esperado']:+.4f}")

    from scipy.stats import beta as beta_dist
    x = np.linspace(
        max(0, min(conv_a/n_a, conv_b/n_b) - 0.05),
        min(1, max(conv_a/n_a, conv_b/n_b) + 0.05), 300)
    fig, ax = plt.subplots(figsize=(9,3))
    ax.fill_between(x, beta_dist.pdf(x, 1+conv_a, 1+n_a-conv_a),
                    alpha=0.6, color="#94a3b8", label=f"{canal_a} (Control)")
    ax.fill_between(x, beta_dist.pdf(x, 1+conv_b, 1+n_b-conv_b),
                    alpha=0.6, color="#2563eb", label=f"{canal_b} (Tratamiento)")
    ax.legend(); ax.set_title("Distribuciones Posteriores Beta", fontweight="bold")
    ax.set_xlabel("Tasa de resolucion")
    sns.despine(); fig.tight_layout()
    st.pyplot(fig); plt.close()
    st.info(f"Recomendacion: {res_bay['recomendacion']}")
