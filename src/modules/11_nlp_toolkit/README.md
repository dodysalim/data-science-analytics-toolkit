# 11 — Kit de Procesamiento de Texto NLP

Pipeline completo de procesamiento de lenguaje natural: limpieza de texto, vectorización TF-IDF, modelado de temas (LDA/LSA), análisis de sentimientos y extracción de n-gramas.

## Autor

**Dody Dueñas**

## Clases

| Clase | Descripción |
|---|---|
| `TextPreprocessor` | Minúsculas, eliminación de URLs/HTML, stopwords, stemming, lematización |
| `NGramExtractor` | Extracción de n-gramas por frecuencia top-k |
| `TFIDFAnalyzer` | Vectorización TF-IDF, términos principales, similitud coseno |
| `TopicModeler` | Modelado de temas LDA o LSA con asignación de tema dominante |
| `LexiconSentimentScorer` | Análisis de sentimientos basado en léxico (sin API requerida) |
| `NLPPipeline` | Pipeline completo de orquestación |

## Uso

```python
from nlp_toolkit import NLPPipeline

textos = df["texto_reseña"].tolist()

pipeline = NLPPipeline(n_topics=5, max_features=3000, topic_method="lda")
resultados = pipeline.run(textos)

print(resultados["topics"])      # Temas LDA
print(resultados["sentiment"])   # Puntuaciones de sentimiento
print(resultados["top_terms"])   # Términos TF-IDF globales
```

## Dependencias

```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
nltk>=3.8.0  # opcional (para lematización/stemming)
```
