"""
Dashboard Interactivo — Data Science Analytics Toolkit
Autor: Dody Dueñas
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

REPO = os.path.dirname(os.path.abspath(__file__))
for mod in ["12_data_profiling","08_feature_engineering","09_model_evaluation",
            "10_data_visualization","11_nlp_toolkit","13_ab_testing","07_time_series"]:
    sys.path.insert(0, os.path.join(REPO, mod))

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DS Toolkit — Dody Dueñas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Fondo principal blanco ── */
[data-testid="stAppViewContainer"] {
    background: #f8fafc;
    color: #1e293b;
}
/* ── Sidebar azul-gris suave ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #2563eb 100%);
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }
/* ── Métricas con tarjetas ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
/* ── Encabezados ── */
h1 { color: #1e3a5f !important; border-bottom: 3px solid #2563eb; padding-bottom: 8px; }
h2 { color: #1e40af !important; }
h3 { color: #2563eb !important; }
/* ── Divider ── */
hr { border-color: #bfdbfe; }
/* ── Botón ── */
.stButton>button {
    background: #2563eb; color: white; border-radius: 8px;
    border: none; padding: 8px 24px; font-weight: 600;
}
.stButton>button:hover { background: #1d4ed8; }
/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x60/1e3a5f/ffffff?text=DS+Toolkit", use_container_width=True)
st.sidebar.title("📊 Navegación")
pagina = st.sidebar.radio("Módulo", [
    "🏠 Inicio",
    "🔍 Perfil de Datos",
    "⚙️ Ingeniería de Características",
    "📈 Series de Tiempo",
    "🤖 Evaluación de Modelos",
    "💬 NLP & Sentimientos",
    "🧪 Pruebas A/B",
])
st.sidebar.markdown("---")
st.sidebar.caption("**Autor:** Dody Dueñas")

# ── Datos compartidos ─────────────────────────────────────────────────────────
@st.cache_data
def generar_datos():
    np.random.seed(42)
    N = 800
    df = pd.DataFrame({
        "fecha_registro": pd.date_range("2021-01-01", periods=N, freq="D"),
        "edad":           np.random.randint(18, 70, N).astype(float),
        "ingresos":       np.random.lognormal(10.5, 0.6, N),
        "puntaje":        np.random.beta(2, 5, N) * 100,
        "deuda":          np.random.exponential(5000, N),
        "segmento":       np.random.choice(["Premium","Estándar","Básico","Prueba"], N),
        "pais":           np.random.choice(["Colombia","México","Argentina","Chile"], N),
        "activo":         np.random.choice([1, 0], N, p=[0.65, 0.35]),
    })
    for col in ["edad","ingresos","pais"]:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
    return df

df = generar_datos()
y  = (df["puntaje"].fillna(50) < 35).astype(int)

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINAS
# ═══════════════════════════════════════════════════════════════════════════════

# ── INICIO ────────────────────────────────────────────────────────────────────
if pagina == "🏠 Inicio":
    st.title("📊 Data Science Analytics Toolkit")
    st.subheader("Autor: **Dody Dueñas**")
    st.markdown("Suite modular de análisis de datos — 7 módulos especializados.")
    st.divider()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Columnas", f"{len(df.columns)}")
    c3.metric("Nulos", f"{df.isna().sum().sum():,}")
    c4.metric("Churn Rate", f"{y.mean():.1%}")

    st.divider()
    st.subheader("Vista rápida del dataset")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Estadísticas descriptivas")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ── PERFIL DE DATOS ───────────────────────────────────────────────────────────
elif pagina == "🔍 Perfil de Datos":
    st.title("🔍 Perfilado de Datos")
    from data_profiling import DatasetProfiler

    with st.spinner("Perfilando dataset..."):
        perfilador = DatasetProfiler()
        reporte = perfilador.profile(df)

    pc = reporte["puntuacion_calidad"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Completitud", f"{pc['completitud']}%")
    c2.metric("Unicidad",    f"{pc['unicidad']}%")
    c3.metric("Consistencia",f"{pc['consistencia']}%")
    c4.metric("Calificación", pc['calificacion'], f"{pc['total']}/100")

    st.divider()
    st.subheader("Mapa de Nulos")
    nulos = df.isna().sum().reset_index()
    nulos.columns = ["Columna","Nulos"]
    nulos["Porcentaje"] = (nulos["Nulos"]/len(df)*100).round(2)
    st.bar_chart(nulos.set_index("Columna")["Porcentaje"])

    st.subheader("Perfil por Columna")
    tabla = perfilador.summary_table(reporte)
    st.dataframe(tabla.round(3), use_container_width=True)

    st.subheader("Distribuciones Numéricas")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    col_sel = st.selectbox("Selecciona columna", num_cols)
    fig, ax = plt.subplots(figsize=(8,3))
    serie = df[col_sel].dropna()
    ax.hist(serie, bins=30, color="#2563eb", alpha=0.7, edgecolor="white")
    ax2 = ax.twinx()
    kde_x = np.linspace(serie.min(), serie.max(), 200)
    kde = stats.gaussian_kde(serie)
    ax2.plot(kde_x, kde(kde_x), color="#f59e0b", lw=2.5)
    ax2.set_yticks([])
    ax.set_title(f"Distribución de {col_sel}", fontsize=12, fontweight="bold")
    sns.despine()
    st.pyplot(fig)
    plt.close()

# ── INGENIERÍA DE CARACTERÍSTICAS ─────────────────────────────────────────────
elif pagina == "⚙️ Ingeniería de Características":
    st.title("⚙️ Ingeniería de Características")
    from feature_engineering import FeatureEngineeringPipeline

    k_best = st.slider("Características a seleccionar (k)", 5, 20, 15)
    poly   = st.slider("Grado polinomial", 1, 3, 2)

    with st.spinner("Aplicando transformaciones..."):
        pipe = FeatureEngineeringPipeline(
            datetime_cols=["fecha_registro"],
            cat_cols=["segmento","pais"],
            num_cols=["edad","ingresos","puntaje","deuda"],
            poly_degree=poly, n_bins=5, k_best=k_best,
            task="classification",
        )
        df_eng = pipe.fit_transform(df.copy(), y)

    c1,c2 = st.columns(2)
    c1.metric("Características originales", df.shape[1])
    c2.metric("Características generadas", df_eng.shape[1])

    scores = pipe.get_feature_scores()
    if scores is not None:
        st.subheader("Puntuacion F por Caracteristica")
        top = scores.head(k_best)
        fig, ax = plt.subplots(figsize=(8, max(3, k_best//3)))
        ax.barh(top["caracteristica"], top["puntuacion"], color="#2563eb", edgecolor="white")
        ax.set_xlabel("Puntuacion F")
        ax.set_title("Top Caracteristicas", fontweight="bold")
        sns.despine()
        st.pyplot(fig)
        plt.close()

    st.subheader("Dataset transformado (primeras filas)")
    st.dataframe(df_eng.head(10).round(3), use_container_width=True)

# ── SERIES DE TIEMPO ──────────────────────────────────────────────────────────
elif pagina == "📈 Series de Tiempo":
    st.title("📈 Análisis de Series de Tiempo")
    from time_series_analyzer import (
        PruebaEstacionariedad, DescompositorSeries,
        PronosticadorARIMA, DetectorAnomaliasSeries
    )

    c1, c2 = st.columns(2)
    pasos  = c1.slider("Meses a pronosticar", 3, 24, 12)
    ventana = c2.slider("Ventana media móvil", 3, 24, 6)

    np.random.seed(42)
    fechas  = pd.date_range("2019-01-01", periods=120, freq="MS")
    serie   = pd.Series(
        np.linspace(100,220,120) + 18*np.sin(2*np.pi*np.arange(120)/12) + np.random.normal(0,6,120),
        index=fechas, name="Ventas")

    # Gráfico serie + media móvil
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(serie.index, serie.values, color="#93c5fd", alpha=0.7, lw=1.5, label="Serie original")
    ax.plot(serie.rolling(ventana).mean().index,
            serie.rolling(ventana).mean().values,
            color="#1d4ed8", lw=2.5, label=f"Media movil {ventana}M")
    ax.legend()
    ax.set_title("Ventas Mensuales", fontweight="bold")
    sns.despine()
    st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estacionariedad")
        prueba = PruebaEstacionariedad()
        adf = prueba.prueba_adf(serie)
        st.json({
            "estadistico": round(adf["estadistico"],4),
            "p_valor": round(adf["p_valor"],6),
            "es_estacionaria": adf["es_estacionaria"],
            "interpretacion": adf["interpretacion"],
        })

    with col2:
        st.subheader(f"Pronóstico ARIMA ({pasos} meses)")
        pronost = PronosticadorARIMA()
        pronost.ajustar(serie, orden=(1,1,1))
        forecast = pronost.pronosticar(pasos=pasos)
        st.line_chart(forecast)

    st.subheader("Descomposición Clásica")
    decomp = DescompositorSeries()
    comp = decomp.extraer_componentes(
        decomp.descomposicion_clasica(serie, periodo=12))
    fig2, axes = plt.subplots(3,1, figsize=(10,6))
    colors = ["#1d4ed8","#0891b2","#dc2626"]
    labels = ["Tendencia","Estacionalidad","Residuo"]
    for ax, col, color, label in zip(axes, comp.columns, colors, labels):
        ax.plot(comp.index, comp[col], color=color, lw=1.5)
        ax.set_title(label, fontweight="bold", fontsize=9)
        ax.tick_params(labelsize=7)
    sns.despine()
    fig2.tight_layout()
    st.pyplot(fig2); plt.close()

# ── EVALUACIÓN DE MODELOS ─────────────────────────────────────────────────────
elif pagina == "🤖 Evaluación de Modelos":
    st.title("🤖 Evaluación de Modelos ML")
    from model_evaluation import ClassificationEvaluator, ModelComparator
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from feature_engineering import FeatureEngineeringPipeline

    with st.spinner("Entrenando modelos..."):
        pipe = FeatureEngineeringPipeline(
            datetime_cols=["fecha_registro"], cat_cols=["segmento","pais"],
            num_cols=["edad","ingresos","puntaje","deuda"],
            poly_degree=2, n_bins=5, k_best=15, task="classification")
        df_eng = pipe.fit_transform(df.copy(), y)
        X = df_eng.select_dtypes(include=np.number).fillna(0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

        rf = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_pred  = rf.predict(X_te)
        y_proba = rf.predict_proba(X_te)

        evaluador = ClassificationEvaluator()
        metricas  = evaluador.evaluate(y_te, y_pred, y_proba)
        cm_data   = evaluador.confusion_matrix_analysis(y_te, y_pred)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Exactitud",  f"{metricas['exactitud']:.3f}")
    c2.metric("F1-Score",   f"{metricas['f1_score']:.3f}")
    c3.metric("ROC-AUC",    f"{metricas.get('roc_auc',0):.3f}")
    c4.metric("Precisión",  f"{metricas['precision']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Curva ROC")
        roc = evaluador.roc_analysis(y_te.values, y_proba)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(roc["fpr"], roc["tpr"], color="#2563eb", lw=2.5,
                label=f"AUC={roc['auc']:.3f}")
        ax.plot([0,1],[0,1],"--",color="#94a3b8",lw=1)
        ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.12, color="#2563eb")
        ax.legend()
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("Curva ROC", fontweight="bold")
        sns.despine()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Matriz de Confusión")
        cm = cm_data["normalized"]
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    linewidths=0.5, ax=ax)
        ax.set_title("Normalizada", fontweight="bold")
        st.pyplot(fig); plt.close()

    st.subheader("Importancia de Características (Top 15)")
    imp = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#2563eb", edgecolor="white")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de Caracteristicas", fontweight="bold")
    sns.despine()
    st.pyplot(fig); plt.close()

    st.subheader("Comparación de Modelos (CV 3-fold)")
    with st.spinner("Comparando..."):
        comp = ModelComparator(task="classification", cv=3)
        comp.add_model("RandomForest",  RandomForestClassifier(n_estimators=50, random_state=42), X, y)
        comp.add_model("GradientBoost", GradientBoostingClassifier(random_state=42), X, y)
        comp.add_model("LogisticReg",   LogisticRegression(max_iter=500, random_state=42), X, y)
    st.dataframe(comp.leaderboard().round(4), use_container_width=True)

# ── NLP ───────────────────────────────────────────────────────────────────────
elif pagina == "💬 NLP & Sentimientos":
    st.title("💬 NLP & Análisis de Sentimientos")
    from nlp_toolkit import NLPPipeline, LexiconSentimentScorer, TextPreprocessor

    texto_usuario = st.text_area("✏️ Escribe tu propio texto para analizar:",
        "Este producto es increíble y muy confiable. Lo recomiendo ampliamente.")

    if st.button("Analizar Sentimiento"):
        scorer = LexiconSentimentScorer()
        res = scorer.puntuar(texto_usuario)
        prep = TextPreprocessor()
        limpio = prep.limpiar(texto_usuario)
        col1, col2, col3 = st.columns(3)
        col1.metric("Palabras Positivas", res["conteo_positivo"])
        col2.metric("Palabras Negativas", res["conteo_negativo"])
        etiqueta = res["etiqueta"].upper()
        color = "🟢" if etiqueta=="POSITIVO" else "🔴" if etiqueta=="NEGATIVO" else "🟡"
        col3.metric("Sentimiento", f"{color} {etiqueta}")
        st.info(f"**Texto limpio:** {limpio}")

    st.divider()
    st.subheader("Análisis de Corpus de Reseñas")
    reseñas = [
        "Este servicio es increíble, rápido y muy confiable. Lo recomiendo.",
        "Pésima experiencia. Todo fue muy lento y el soporte fue terrible.",
        "Producto bueno, precio justo. Funciona como se esperaba.",
        "Fantástico! La mejor plataforma. Excelente atención al cliente.",
        "Horrible. El sistema falla constantemente y el servicio es muy malo.",
        "Buena relación calidad-precio. Interfaz simple y fácil de usar.",
        "No lo recomiendo. Muy complicado y con muchos errores.",
        "Genial experiencia. El equipo es brillante y el producto es perfecto.",
        "Regular. Tiene algunas ventajas pero también muchos problemas técnicos.",
        "Increíble! Rápida, confiable y con excelente soporte técnico.",
    ] * 8

    with st.spinner("Procesando corpus NLP..."):
        pipeline_nlp = NLPPipeline(n_temas=3, max_features=300)
        res_nlp = pipeline_nlp.ejecutar(reseñas)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de Sentimientos")
        dist = res_nlp["sentimientos"]["etiqueta"].value_counts()
        fig, ax = plt.subplots(figsize=(4,4))
        colores = {"positivo":"#22c55e","neutral":"#f59e0b","negativo":"#ef4444"}
        ax.pie(dist.values, labels=dist.index, autopct="%1.1f%%",
               colors=[colores.get(l,"#2563eb") for l in dist.index],
               wedgeprops={"width":0.5,"edgecolor":"white"})
        ax.set_title("Sentimientos", fontweight="bold")
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Temas Descubiertos (LDA)")
        st.dataframe(res_nlp["temas"], use_container_width=True)

    st.subheader("Top Bigramas")
    st.dataframe(res_nlp["bigramas"].head(10), use_container_width=True)

    st.subheader("Términos TF-IDF Globales")
    st.dataframe(res_nlp["terminos_top"].head(10), use_container_width=True)

# ── A/B TESTING ───────────────────────────────────────────────────────────────
elif pagina == "🧪 Pruebas A/B":
    st.title("🧪 Pruebas A/B & Análisis de Experimentos")
    from ab_testing import SampleSizeCalculator, FrequentistABTester, BayesianABTester

    st.subheader("⚙️ Configuración del Experimento")
    col1, col2 = st.columns(2)
    with col1:
        baseline    = st.slider("Tasa de conversión baseline (%)", 1, 30, 8) / 100
        mde         = st.slider("Efecto mínimo detectable (%)", 5, 50, 15) / 100
        potencia    = st.slider("Potencia estadística (%)", 70, 95, 80) / 100
    with col2:
        conv_ctrl   = st.number_input("Conversiones Control",   value=800)
        n_ctrl      = st.number_input("N Control",              value=10000)
        conv_trat   = st.number_input("Conversiones Tratamiento", value=940)
        n_trat      = st.number_input("N Tratamiento",          value=10000)

    calc = SampleSizeCalculator()
    plan = calc.for_proportions(baseline, mde, power=potencia)

    st.divider()
    st.subheader("📐 Tamaño de Muestra Requerido")
    c1,c2,c3 = st.columns(3)
    c1.metric("N por grupo",  f"{plan['n_por_grupo']:,}")
    c2.metric("Total N",      f"{plan['total_n']:,}")
    c3.metric("Tasa esperada tratamiento", f"{plan['tasa_tratamiento']:.2%}")

    st.divider()
    tester = FrequentistABTester()
    res_z  = tester.proportion_test(int(conv_ctrl), int(n_ctrl), int(conv_trat), int(n_trat))

    st.subheader("📊 Resultados Frecuentistas")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tasa Control",     f"{res_z['tasa_control']:.2%}")
    c2.metric("Tasa Tratamiento", f"{res_z['tasa_tratamiento']:.2%}")
    c3.metric("Incremento",       f"{res_z['incremento_relativo_pct']:+.2f}%")
    c4.metric("p-valor",          f"{res_z['p_valor']:.5f}")

    sig = res_z["es_significativo"]
    if sig:
        st.success(f"✅ **SIGNIFICATIVO** — {res_z['recomendacion']}")
    else:
        st.warning(f"⚠️ **NO SIGNIFICATIVO** — {res_z['recomendacion']}")

    st.info(f"**Z-stat:** {res_z['estadistico_z']:.4f} | **IC 95%:** {res_z['intervalo_confianza_95pct']}")

    st.subheader("🎲 Análisis Bayesiano")
    bay = BayesianABTester()
    res_bay = bay.analyze(int(conv_ctrl), int(n_ctrl), int(conv_trat), int(n_trat))

    c1,c2 = st.columns(2)
    c1.metric("P(Tratamiento gana)", f"{res_bay['prob_tratamiento_gana']:.1%}")
    c2.metric("Incremento esperado", f"{res_bay['media_incremento_esperado']:+.4f}")

    # Distribuciones Beta posteriores
    fig, ax = plt.subplots(figsize=(8,3))
    x = np.linspace(0.05, 0.15, 300)
    from scipy.stats import beta as beta_dist
    ac = 1 + int(conv_ctrl); bc = 1 + int(n_ctrl - conv_ctrl)
    at = 1 + int(conv_trat); bt = 1 + int(n_trat - conv_trat)
    ax.fill_between(x, beta_dist.pdf(x, ac, bc), alpha=0.6, color="#94a3b8", label="Control")
    ax.fill_between(x, beta_dist.pdf(x, at, bt), alpha=0.6, color="#2563eb", label="Tratamiento")
    ax.legend()
    ax.set_title("Distribuciones Posteriores Beta", fontweight="bold")
    ax.set_xlabel("Tasa de conversion")
    sns.despine()
    st.pyplot(fig); plt.close()

    st.info(f"**Recomendación Bayesiana:** {res_bay['recomendacion']}")
