# 11 — NLP Text Processing Toolkit

End-to-end natural language processing pipeline: text cleaning, TF-IDF vectorization, topic modeling (LDA/LSA), sentiment scoring, and n-gram analysis.

## Classes

| Class | Description |
|---|---|
| `TextPreprocessor` | Lowercase, URL/HTML removal, stopwords, stemming, lemmatization |
| `NGramExtractor` | Top-k n-gram frequency extraction |
| `TFIDFAnalyzer` | TF-IDF vectorization, top terms, cosine similarity |
| `TopicModeler` | LDA or LSA topic modeling with dominant topic assignment |
| `LexiconSentimentScorer` | Lexicon-based sentiment analysis (no API required) |
| `NLPPipeline` | Full orchestration pipeline |

## Usage

```python
from nlp_toolkit import NLPPipeline

texts = df["review_text"].tolist()

pipeline = NLPPipeline(n_topics=5, max_features=3000, topic_method="lda")
results = pipeline.run(texts)

print(results["topics"])          # LDA topics
print(results["sentiment"])       # Sentiment scores
print(results["top_terms"])       # Global TF-IDF terms
```

## Requirements

```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
nltk>=3.8.0  # optional (for lemmatization/stemming)
```
