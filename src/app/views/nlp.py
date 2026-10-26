import streamlit as st
from nlp_toolkit import NLPPipeline, LexiconSentimentScorer

from .components import fig_pie, show

def render_nlp(df_raw, df):
    st.title("💬 Analisis de Texto — NLP Real")

    st.subheader("Analizar texto en vivo")
    texto_in = st.text_area("Escribe o pega un mensaje de soporte:",
        "Hello, my SSO is not working as expected. The service has been down for 3 days.")

    if st.button("Analizar Sentimiento"):
        scorer = LexiconSentimentScorer()
        res = scorer.puntuar(texto_in)
        col1, col2, col3 = st.columns(3)
        col1.metric("Palabras Positivas", res["conteo_positivo"])
        col2.metric("Palabras Negativas", res["conteo_negativo"])
        emoji = "🟢" if res["etiqueta"]=="positivo" else "🔴" if res["etiqueta"]=="negativo" else "🟡"
        col3.metric("Sentimiento", f"{emoji} {res['etiqueta'].upper()}")

    st.divider()
    st.subheader("Analisis del corpus de mensajes reales")

    n_nlp = st.slider("Mensajes a analizar", 100, 2000, 500, step=100)
    filtro_sent = st.selectbox("Filtrar por sentimiento", ["Todos","negative","neutral","positive"])

    textos_df = df_raw[df_raw["role"]=="customer"].dropna(subset=["text"])
    if filtro_sent != "Todos":
        textos_df = textos_df[textos_df["overall_sentiment"] == filtro_sent]
    textos = textos_df["text"].head(n_nlp).tolist()

    with st.spinner(f"Procesando {len(textos)} mensajes reales..."):
        pipeline_nlp = NLPPipeline(n_temas=4, max_features=500)
        res_nlp = pipeline_nlp.ejecutar(textos)

    nlp_colors = {"negativo": "#FF8A80", "neutral": "#F8D66D", "positivo": "#75E6CE",
                  "negative": "#FF8A80", "positive": "#75E6CE"}
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentimientos detectados por NLP")
        dist = res_nlp["sentimientos"]["etiqueta"].value_counts()
        show(fig_pie(dist, "Sentimiento (léxico)", colors=[nlp_colors.get(l, "#88B7FF") for l in dist.index]))
    with col2:
        st.subheader("Temas descubiertos (LDA)")
        st.dataframe(res_nlp["temas"], use_container_width=True)

    st.subheader("Top Terminos TF-IDF")
    st.dataframe(res_nlp["terminos_top"].head(15), use_container_width=True)

    st.subheader("Top Bigramas mas frecuentes")
    st.dataframe(res_nlp["bigramas"].head(15), use_container_width=True)
