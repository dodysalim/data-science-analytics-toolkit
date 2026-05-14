"""
Kit de Procesamiento de Texto NLP
=====================================
Pipeline completo de procesamiento de lenguaje natural:
- Preprocesamiento y normalización de texto
- Vectorización TF-IDF y de conteo
- Extracción de N-gramas
- Modelado de temas (LDA)
- Puntuación de sentimientos por léxico
- Cálculo de similitud entre textos

Autor: Dody Dueñas
"""

import re
import string
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_DISPONIBLE = True
except ImportError:
    NLTK_DISPONIBLE = False


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESADOR DE TEXTO
# ──────────────────────────────────────────────────────────────────────────────

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Limpia y normaliza datos de texto sin procesar.

    Operaciones (todas configurables):
      - Conversión a minúsculas
      - Eliminación de URLs
      - Eliminación de etiquetas HTML
      - Eliminación de puntuación
      - Eliminación de números
      - Eliminación de stopwords
      - Stemming / Lematización (requiere NLTK)
      - Normalización de espacios en blanco
    """

    STOPWORDS_ESPAÑOL = {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o",
        "pero", "en", "de", "del", "al", "con", "por", "para", "que",
        "se", "no", "si", "más", "es", "son", "está", "están", "fue",
        "ser", "ha", "han", "ya", "su", "sus", "le", "les", "lo", "me",
        "te", "nos", "vos", "yo", "él", "ella", "ello", "este", "esta",
        "esto", "ese", "esa", "eso", "aquel", "como", "también", "muy",
        "bien", "así", "cuando", "donde", "quien", "cual", "todo", "todos",
    }

    STOPWORDS_INGLES = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "is", "it", "its", "this", "that",
        "was", "are", "be", "been", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "not",
        "no", "so", "if", "as", "from", "up", "out", "then", "than",
        "also", "into", "about", "what", "which", "who", "there", "their",
        "they", "we", "you", "he", "she", "i", "me", "my", "our", "all",
    }

    def __init__(
        self,
        minusculas: bool = True,
        eliminar_urls: bool = True,
        eliminar_html: bool = True,
        eliminar_puntuacion: bool = True,
        eliminar_numeros: bool = False,
        eliminar_stopwords: bool = True,
        longitud_minima_palabra: int = 2,
        lematizar: bool = False,
        aplicar_stemming: bool = False,
        stopwords_personalizadas: Optional[List[str]] = None,
        idioma: str = "es",
    ):
        self.minusculas = minusculas
        self.eliminar_urls = eliminar_urls
        self.eliminar_html = eliminar_html
        self.eliminar_puntuacion = eliminar_puntuacion
        self.eliminar_numeros = eliminar_numeros
        self.eliminar_stopwords = eliminar_stopwords
        self.longitud_minima_palabra = longitud_minima_palabra
        self.lematizar = lematizar
        self.aplicar_stemming = aplicar_stemming
        self.stopwords_personalizadas = set(stopwords_personalizadas or [])
        self.idioma = idioma

        base_sw = self.STOPWORDS_ESPAÑOL if idioma == "es" else self.STOPWORDS_INGLES
        self._stopwords = base_sw | self.stopwords_personalizadas
        self._lematizador = WordNetLemmatizer() if (lematizar and NLTK_DISPONIBLE) else None
        self._stemmer = PorterStemmer() if (aplicar_stemming and NLTK_DISPONIBLE) else None

    def limpiar(self, texto: str) -> str:
        """Aplica todos los pasos de limpieza a una cadena individual."""
        if not isinstance(texto, str):
            return ""
        if self.minusculas:
            texto = texto.lower()
        if self.eliminar_html:
            texto = re.sub(r"<[^>]+>", " ", texto)
        if self.eliminar_urls:
            texto = re.sub(r"http\S+|www\.\S+", " ", texto)
        if self.eliminar_puntuacion:
            texto = texto.translate(str.maketrans("", "", string.punctuation))
        if self.eliminar_numeros:
            texto = re.sub(r"\d+", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()

        palabras = texto.split()
        if self.eliminar_stopwords:
            palabras = [p for p in palabras if p not in self._stopwords]
        palabras = [p for p in palabras if len(p) >= self.longitud_minima_palabra]

        if self._lematizador:
            palabras = [self._lematizador.lemmatize(p) for p in palabras]
        elif self._stemmer:
            palabras = [self._stemmer.stem(p) for p in palabras]

        return " ".join(palabras)

    def fit(self, X, y=None) -> "TextPreprocessor":
        return self

    def transform(self, X: Union[pd.Series, List[str]]) -> pd.Series:
        """Limpia una Serie o lista de documentos."""
        serie = pd.Series(X) if not isinstance(X, pd.Series) else X
        return serie.apply(self.limpiar)


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTOR DE N-GRAMAS
# ──────────────────────────────────────────────────────────────────────────────

class NGramExtractor:
    """Extrae y clasifica n-gramas de un corpus de texto."""

    def __init__(self, n: int = 2, top_k: int = 20):
        self.n = n
        self.top_k = top_k

    def extraer(self, corpus: Union[pd.Series, List[str]]) -> pd.DataFrame:
        """
        Extrae los top-k n-gramas de un corpus.

        Parámetros
        ----------
        corpus : list o pd.Series de cadenas de texto

        Retorna
        -------
        pd.DataFrame con columnas: ngrama, conteo, frecuencia
        """
        vectorizador = CountVectorizer(ngram_range=(self.n, self.n))
        X = vectorizador.fit_transform(corpus)
        conteos = X.sum(axis=0).A1
        vocabulario = vectorizador.get_feature_names_out()
        df = pd.DataFrame({"ngrama": vocabulario, "conteo": conteos})
        total = conteos.sum()
        df["frecuencia"] = df["conteo"] / total
        return df.sort_values("conteo", ascending=False).head(self.top_k).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# ANALIZADOR TF-IDF
# ──────────────────────────────────────────────────────────────────────────────

class TFIDFAnalyzer:
    """
    Vectorización TF-IDF con extracción de términos principales y similitud entre documentos.
    """

    def __init__(
        self,
        max_features: int = 5000,
        rango_ngrama: Tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.vectorizador = TfidfVectorizer(
            max_features=max_features,
            ngram_range=rango_ngrama,
            min_df=min_df,
            max_df=max_df,
        )
        self._matriz = None

    def fit_transform(self, corpus: Union[pd.Series, List[str]]) -> np.ndarray:
        """Ajusta TF-IDF y retorna la matriz documento-término."""
        self._matriz = self.vectorizador.fit_transform(corpus)
        return self._matriz

    def terminos_top_por_documento(self, n: int = 10) -> pd.DataFrame:
        """
        Retorna los n términos TF-IDF más importantes por documento.

        Retorna
        -------
        pd.DataFrame con id_doc y lista de términos principales.
        """
        if self._matriz is None:
            raise ValueError("Llame a fit_transform primero.")
        nombres_caracteristicas = self.vectorizador.get_feature_names_out()
        registros = []
        for i, fila in enumerate(self._matriz):
            fila_densa = fila.toarray().flatten()
            idx_top = fila_densa.argsort()[-n:][::-1]
            terminos_top = [(nombres_caracteristicas[j], round(fila_densa[j], 4)) for j in idx_top]
            registros.append({"id_doc": i, "terminos_top": terminos_top})
        return pd.DataFrame(registros)

    def terminos_globales_top(self, n: int = 30) -> pd.DataFrame:
        """Retorna los términos globalmente más importantes por puntuación TF-IDF promedio."""
        if self._matriz is None:
            raise ValueError("Llame a fit_transform primero.")
        puntuaciones_prom = self._matriz.mean(axis=0).A1
        nombres_caracteristicas = self.vectorizador.get_feature_names_out()
        return (
            pd.DataFrame({"termino": nombres_caracteristicas, "tfidf_prom": puntuaciones_prom})
            .sort_values("tfidf_prom", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def similitud_documentos(self, idx_a: int, idx_b: int) -> float:
        """Calcula la similitud coseno entre dos documentos."""
        if self._matriz is None:
            raise ValueError("Llame a fit_transform primero.")
        a = self._matriz[idx_a]
        b = self._matriz[idx_b]
        return float(cosine_similarity(a, b)[0, 0])

    def documentos_similares(
        self, idx_consulta: int, top_n: int = 5
    ) -> pd.DataFrame:
        """Encuentra los top-n documentos más similares a uno dado."""
        if self._matriz is None:
            raise ValueError("Llame a fit_transform primero.")
        vec_consulta = self._matriz[idx_consulta]
        similitudes = cosine_similarity(vec_consulta, self._matriz).flatten()
        similitudes[idx_consulta] = -1  # excluir el propio documento
        indices_top = similitudes.argsort()[-top_n:][::-1]
        return pd.DataFrame({
            "id_doc": indices_top,
            "similitud": similitudes[indices_top].round(4),
        })


# ──────────────────────────────────────────────────────────────────────────────
# MODELADOR DE TEMAS
# ──────────────────────────────────────────────────────────────────────────────

class TopicModeler:
    """
    Descubre temas latentes usando LDA o LSA (TruncatedSVD).

    Soporta:
      - LDA  : Asignación Latente de Dirichlet
      - LSA  : Análisis Semántico Latente (vía TruncatedSVD)
    """

    def __init__(
        self,
        n_temas: int = 10,
        metodo: str = "lda",
        max_features: int = 3000,
        n_palabras_top: int = 10,
        semilla_aleatoria: int = 42,
    ):
        self.n_temas = n_temas
        self.metodo = metodo
        self.max_features = max_features
        self.n_palabras_top = n_palabras_top
        self.semilla_aleatoria = semilla_aleatoria
        self._vectorizador = None
        self._modelo = None

    def ajustar(self, corpus: Union[pd.Series, List[str]]) -> "TopicModeler":
        """Ajusta el modelo de temas sobre un corpus de documentos."""
        self._vectorizador = CountVectorizer(
            max_features=self.max_features, stop_words="english"
        )
        X = self._vectorizador.fit_transform(corpus)

        if self.metodo == "lda":
            self._modelo = LatentDirichletAllocation(
                n_components=self.n_temas,
                random_state=self.semilla_aleatoria,
                max_iter=20,
            )
        elif self.metodo == "lsa":
            self._modelo = TruncatedSVD(
                n_components=self.n_temas, random_state=self.semilla_aleatoria
            )
        else:
            raise ValueError("El método debe ser 'lda' o 'lsa'.")
        self._modelo.fit(X)
        return self

    def obtener_temas(self) -> pd.DataFrame:
        """
        Retorna las palabras principales por tema.

        Retorna
        -------
        pd.DataFrame con columnas: id_tema, palabras_top
        """
        if self._modelo is None:
            raise ValueError("Llame a ajustar() primero.")
        nombres_caracteristicas = self._vectorizador.get_feature_names_out()
        registros = []
        for idx_tema, tema in enumerate(self._modelo.components_):
            indices_top = tema.argsort()[-self.n_palabras_top:][::-1]
            palabras_top = [nombres_caracteristicas[i] for i in indices_top]
            registros.append({
                "id_tema": idx_tema,
                "palabras_top": ", ".join(palabras_top),
            })
        return pd.DataFrame(registros)

    def transformar(self, corpus: Union[pd.Series, List[str]]) -> np.ndarray:
        """Obtiene la distribución documento-tema para nuevos documentos."""
        if self._modelo is None:
            raise ValueError("Llame a ajustar() primero.")
        X = self._vectorizador.transform(corpus)
        return self._modelo.transform(X)

    def tema_dominante(self, corpus: Union[pd.Series, List[str]]) -> pd.Series:
        """Retorna el índice del tema dominante para cada documento."""
        distribucion = self.transformar(corpus)
        return pd.Series(distribucion.argmax(axis=1), name="tema_dominante")


# ──────────────────────────────────────────────────────────────────────────────
# ANALIZADOR DE SENTIMIENTOS POR LÉXICO
# ──────────────────────────────────────────────────────────────────────────────

class LexiconSentimentScorer:
    """
    Análisis de sentimientos simple basado en léxico.
    Cuenta palabras positivas y negativas usando un léxico semilla incorporado.
    No requiere API ni modelo externo.
    """

    PALABRAS_POSITIVAS = {
        "bueno", "excelente", "fantástico", "increíble", "maravilloso",
        "perfecto", "genial", "satisfecho", "recomiendo", "impresionante",
        "brillante", "útil", "eficiente", "rápido", "fácil", "limpio",
        "simple", "confiable", "disfruto", "hermoso", "good", "great",
        "excellent", "awesome", "fantastic", "love", "best", "happy",
        "perfect", "amazing", "wonderful", "helpful", "efficient", "fast",
        "reliable", "recommend", "impressive", "brilliant", "smooth",
    }

    PALABRAS_NEGATIVAS = {
        "malo", "terrible", "horrible", "odio", "pésimo", "lento",
        "difícil", "roto", "error", "fallo", "problema", "molesto",
        "inútil", "deficiente", "decepcionante", "feo", "confuso",
        "poco confiable", "desperdicio", "complicado", "demora",
        "bad", "terrible", "awful", "hate", "worst", "horrible", "slow",
        "difficult", "broken", "error", "fail", "problem", "issue",
        "frustrating", "annoying", "useless", "poor", "disappointing",
        "ugly", "confusing", "crash", "unreliable", "waste", "expensive",
    }

    def puntuar(self, texto: str) -> Dict[str, Any]:
        """
        Calcula la puntuación de sentimiento para un texto individual.

        Retorna
        -------
        dict con conteo_positivo, conteo_negativo, puntaje_neto, etiqueta
        """
        palabras = set(texto.lower().split())
        positivas = len(palabras & self.PALABRAS_POSITIVAS)
        negativas = len(palabras & self.PALABRAS_NEGATIVAS)
        neto = positivas - negativas
        etiqueta = "positivo" if neto > 0 else "negativo" if neto < 0 else "neutral"
        return {
            "conteo_positivo": positivas,
            "conteo_negativo": negativas,
            "puntaje_neto": neto,
            "etiqueta": etiqueta,
        }

    def puntuar_corpus(self, corpus: Union[pd.Series, List[str]]) -> pd.DataFrame:
        """Puntúa un corpus completo de textos."""
        serie = pd.Series(corpus)
        puntuaciones = serie.apply(self.puntuar)
        return pd.DataFrame(puntuaciones.tolist())


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE NLP
# ──────────────────────────────────────────────────────────────────────────────

class NLPPipeline:
    """
    Pipeline NLP de extremo a extremo:
      1. Limpieza de texto
      2. Extracción de N-gramas
      3. Cálculo TF-IDF
      4. Modelado de temas
      5. Análisis de sentimientos
    """

    def __init__(
        self,
        n_temas: int = 5,
        max_features: int = 2000,
        metodo_temas: str = "lda",
    ):
        self.preprocesador = TextPreprocessor()
        self.extractor_ngramas = NGramExtractor(n=2, top_k=20)
        self.analizador_tfidf = TFIDFAnalyzer(max_features=max_features)
        self.modelador_temas = TopicModeler(
            n_temas=n_temas, metodo=metodo_temas, max_features=max_features
        )
        self.analizador_sentimientos = LexiconSentimentScorer()

    def ejecutar(self, corpus: Union[pd.Series, List[str]]) -> Dict[str, Any]:
        """
        Ejecuta todos los pasos del pipeline.

        Retorna
        -------
        dict con: 'corpus_limpio', 'bigramas', 'terminos_top', 'temas',
                  'temas_dominantes', 'sentimientos'
        """
        corpus = pd.Series(corpus)
        resultados = {}

        # 1. Limpiar
        limpio = self.preprocesador.transform(corpus)
        resultados["corpus_limpio"] = limpio

        # 2. N-gramas
        resultados["bigramas"] = self.extractor_ngramas.extraer(limpio)

        # 3. TF-IDF
        self.analizador_tfidf.fit_transform(limpio)
        resultados["terminos_top"] = self.analizador_tfidf.terminos_globales_top(n=20)

        # 4. Temas
        self.modelador_temas.ajustar(limpio)
        resultados["temas"] = self.modelador_temas.obtener_temas()
        resultados["temas_dominantes"] = self.modelador_temas.tema_dominante(limpio)

        # 5. Sentimientos
        resultados["sentimientos"] = self.analizador_sentimientos.puntuar_corpus(corpus)

        return resultados


# ──────────────────────────────────────────────────────────────────────────────
# Demostración
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    textos_muestra = [
        "Este producto es increíble! Excelente calidad y entrega rápida. Me encanta.",
        "Terrible experiencia. El artículo llegó roto y el servicio fue horrible.",
        "Producto decente, nada especial. Funciona como se esperaba.",
        "Absolutamente fantástico! La mejor compra que hice este año. Lo recomiendo.",
        "Muy lento y poco confiable. La aplicación falla constantemente. Frustrante!",
        "Buena relación calidad-precio. Interfaz simple e intuitiva.",
        "No compren esto. Pésima calidad. Un completo desperdicio de dinero.",
        "Bastante bueno en general. Algunos problemas menores pero nada grave.",
    ] * 10  # ampliar corpus para modelado de temas

    pipeline = NLPPipeline(n_temas=3, max_features=500)
    resultados = pipeline.ejecutar(textos_muestra)

    print("=== Bigramas más frecuentes ===")
    print(resultados["bigramas"].head())

    print("\n=== Términos TF-IDF Principales ===")
    print(resultados["terminos_top"].head(10))

    print("\n=== Temas Descubiertos ===")
    print(resultados["temas"])

    print("\n=== Distribución de Sentimientos ===")
    print(resultados["sentimientos"]["etiqueta"].value_counts())
