"""
NLP Text Processing Toolkit
==============================
End-to-end natural language processing pipeline:
- Text preprocessing and normalization
- TF-IDF and Count vectorization
- N-gram extraction
- Topic modeling (LDA)
- Sentiment lexicon scoring
- Named entity recognition helpers
- Text similarity computation

Author: Data Science Analytics Toolkit
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
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# TEXT PREPROCESSOR
# ──────────────────────────────────────────────────────────────────────────────

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Clean and normalize raw text data.

    Operations (all configurable):
      - Lowercase conversion
      - URL removal
      - HTML tag stripping
      - Punctuation removal
      - Number removal
      - Stopword removal (requires NLTK)
      - Stemming / Lemmatization (requires NLTK)
      - Whitespace normalization
    """

    ENGLISH_STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "is", "it", "its", "this", "that",
        "was", "are", "be", "been", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "not", "no", "so", "if", "as", "from", "up", "out", "then",
        "than", "also", "into", "about", "what", "which", "who", "whom",
        "there", "their", "they", "we", "you", "he", "she", "i", "me",
        "my", "our", "your", "his", "her", "them", "us", "all",
    }

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        min_word_length: int = 2,
        lemmatize: bool = False,
        stem: bool = False,
        custom_stopwords: Optional[List[str]] = None,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.lemmatize = lemmatize
        self.stem = stem
        self.custom_stopwords = set(custom_stopwords or [])

        self._stopwords = self.ENGLISH_STOPWORDS | self.custom_stopwords
        self._lemmatizer = WordNetLemmatizer() if (lemmatize and NLTK_AVAILABLE) else None
        self._stemmer = PorterStemmer() if (stem and NLTK_AVAILABLE) else None

    def clean(self, text: str) -> str:
        """Apply all cleaning steps to a single string."""
        if not isinstance(text, str):
            return ""
        if self.lowercase:
            text = text.lower()
        if self.remove_html:
            text = re.sub(r"<[^>]+>", " ", text)
        if self.remove_urls:
            text = re.sub(r"http\S+|www\.\S+", " ", text)
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Token-level operations
        words = text.split()
        if self.remove_stopwords:
            words = [w for w in words if w not in self._stopwords]
        words = [w for w in words if len(w) >= self.min_word_length]

        if self._lemmatizer:
            words = [self._lemmatizer.lemmatize(w) for w in words]
        elif self._stemmer:
            words = [self._stemmer.stem(w) for w in words]

        return " ".join(words)

    def fit(self, X, y=None) -> "TextPreprocessor":
        return self

    def transform(self, X: Union[pd.Series, List[str]]) -> pd.Series:
        """Clean a Series or list of documents."""
        series = pd.Series(X) if not isinstance(X, pd.Series) else X
        return series.apply(self.clean)


# ──────────────────────────────────────────────────────────────────────────────
# NGRAM EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

class NGramExtractor:
    """Extract and rank n-grams from a text corpus."""

    def __init__(self, n: int = 2, top_k: int = 20):
        self.n = n
        self.top_k = top_k

    def extract(self, corpus: Union[pd.Series, List[str]]) -> pd.DataFrame:
        """
        Extract top-k n-grams from a corpus.

        Parameters
        ----------
        corpus : list or pd.Series of strings

        Returns
        -------
        pd.DataFrame with columns: ngram, count, frequency
        """
        vectorizer = CountVectorizer(ngram_range=(self.n, self.n))
        X = vectorizer.fit_transform(corpus)
        counts = X.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        df = pd.DataFrame({"ngram": vocab, "count": counts})
        total = counts.sum()
        df["frequency"] = df["count"] / total
        return df.sort_values("count", ascending=False).head(self.top_k).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF ANALYZER
# ──────────────────────────────────────────────────────────────────────────────

class TFIDFAnalyzer:
    """
    TF-IDF vectorization with top-term extraction and document similarity.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        self._matrix = None

    def fit_transform(self, corpus: Union[pd.Series, List[str]]) -> np.ndarray:
        """Fit TF-IDF and return document-term matrix."""
        self._matrix = self.vectorizer.fit_transform(corpus)
        return self._matrix

    def top_terms_per_doc(self, n: int = 10) -> pd.DataFrame:
        """
        Return the top-n TF-IDF terms for each document.

        Returns
        -------
        pd.DataFrame with doc_id and list of top terms.
        """
        if self._matrix is None:
            raise ValueError("Call fit_transform first.")
        feature_names = self.vectorizer.get_feature_names_out()
        records = []
        for i, row in enumerate(self._matrix):
            row_dense = row.toarray().flatten()
            top_idx = row_dense.argsort()[-n:][::-1]
            top_terms = [(feature_names[j], round(row_dense[j], 4)) for j in top_idx]
            records.append({"doc_id": i, "top_terms": top_terms})
        return pd.DataFrame(records)

    def global_top_terms(self, n: int = 30) -> pd.DataFrame:
        """Return globally most important terms by average TF-IDF score."""
        if self._matrix is None:
            raise ValueError("Call fit_transform first.")
        avg_scores = self._matrix.mean(axis=0).A1
        feature_names = self.vectorizer.get_feature_names_out()
        return (
            pd.DataFrame({"term": feature_names, "avg_tfidf": avg_scores})
            .sort_values("avg_tfidf", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def document_similarity(self, doc_idx_a: int, doc_idx_b: int) -> float:
        """Compute cosine similarity between two documents."""
        if self._matrix is None:
            raise ValueError("Call fit_transform first.")
        a = self._matrix[doc_idx_a]
        b = self._matrix[doc_idx_b]
        return float(cosine_similarity(a, b)[0, 0])

    def find_similar_documents(
        self, query_idx: int, top_n: int = 5
    ) -> pd.DataFrame:
        """Find the top-n most similar documents to a given document."""
        if self._matrix is None:
            raise ValueError("Call fit_transform first.")
        query_vec = self._matrix[query_idx]
        similarities = cosine_similarity(query_vec, self._matrix).flatten()
        similarities[query_idx] = -1  # exclude self
        top_indices = similarities.argsort()[-top_n:][::-1]
        return pd.DataFrame({
            "doc_id": top_indices,
            "similarity": similarities[top_indices].round(4),
        })


# ──────────────────────────────────────────────────────────────────────────────
# TOPIC MODELER
# ──────────────────────────────────────────────────────────────────────────────

class TopicModeler:
    """
    Discover latent topics using LDA or LSA (TruncatedSVD).

    Supports:
      - LDA  : Latent Dirichlet Allocation
      - LSA  : Latent Semantic Analysis (via TruncatedSVD)
    """

    def __init__(
        self,
        n_topics: int = 10,
        method: str = "lda",
        max_features: int = 3000,
        n_top_words: int = 10,
        random_state: int = 42,
    ):
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features
        self.n_top_words = n_top_words
        self.random_state = random_state
        self._vectorizer = None
        self._model = None

    def fit(self, corpus: Union[pd.Series, List[str]]) -> "TopicModeler":
        """Fit topic model on a corpus of documents."""
        self._vectorizer = CountVectorizer(
            max_features=self.max_features, stop_words="english"
        )
        X = self._vectorizer.fit_transform(corpus)

        if self.method == "lda":
            self._model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=20,
            )
        elif self.method == "lsa":
            self._model = TruncatedSVD(
                n_components=self.n_topics, random_state=self.random_state
            )
        else:
            raise ValueError("method must be 'lda' or 'lsa'.")
        self._model.fit(X)
        return self

    def get_topics(self) -> pd.DataFrame:
        """
        Return top words per topic.

        Returns
        -------
        pd.DataFrame with columns: topic_id, top_words
        """
        if self._model is None:
            raise ValueError("Call fit() first.")
        feature_names = self._vectorizer.get_feature_names_out()
        records = []
        for topic_idx, topic in enumerate(self._model.components_):
            top_indices = topic.argsort()[-self.n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            records.append({
                "topic_id": topic_idx,
                "top_words": ", ".join(top_words),
            })
        return pd.DataFrame(records)

    def transform(self, corpus: Union[pd.Series, List[str]]) -> np.ndarray:
        """Get document-topic distribution for new documents."""
        if self._model is None:
            raise ValueError("Call fit() first.")
        X = self._vectorizer.transform(corpus)
        return self._model.transform(X)

    def dominant_topic(self, corpus: Union[pd.Series, List[str]]) -> pd.Series:
        """Return the dominant topic index for each document."""
        dist = self.transform(corpus)
        return pd.Series(dist.argmax(axis=1), name="dominant_topic")


# ──────────────────────────────────────────────────────────────────────────────
# SENTIMENT SCORER (lexicon-based, no external model needed)
# ──────────────────────────────────────────────────────────────────────────────

class LexiconSentimentScorer:
    """
    Simple lexicon-based sentiment scoring.
    Counts positive and negative words using a built-in seed lexicon.
    No external API or model required.
    """

    POSITIVE_WORDS = {
        "good", "great", "excellent", "awesome", "fantastic", "love",
        "best", "happy", "satisfied", "perfect", "amazing", "wonderful",
        "positive", "nice", "superb", "outstanding", "brilliant", "helpful",
        "efficient", "easy", "fast", "reliable", "recommend", "impressive",
        "beautiful", "clean", "simple", "intuitive", "smooth", "enjoy",
    }

    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "hate", "worst", "horrible", "slow",
        "difficult", "broken", "error", "fail", "problem", "issue", "bug",
        "frustrating", "annoying", "useless", "poor", "disappointing",
        "ugly", "confusing", "crash", "unreliable", "waste", "expensive",
        "complicated", "wrong", "missing", "broken", "delay", "laggy",
    }

    def score(self, text: str) -> Dict[str, Any]:
        """
        Compute sentiment scores for a single text.

        Returns
        -------
        dict with positive_count, negative_count, net_score, label
        """
        words = set(text.lower().split())
        pos = len(words & self.POSITIVE_WORDS)
        neg = len(words & self.NEGATIVE_WORDS)
        net = pos - neg
        label = "positive" if net > 0 else "negative" if net < 0 else "neutral"
        return {
            "positive_count": pos,
            "negative_count": neg,
            "net_score": net,
            "label": label,
        }

    def score_corpus(self, corpus: Union[pd.Series, List[str]]) -> pd.DataFrame:
        """Score an entire corpus."""
        series = pd.Series(corpus)
        scores = series.apply(self.score)
        return pd.DataFrame(scores.tolist())


# ──────────────────────────────────────────────────────────────────────────────
# NLP PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

class NLPPipeline:
    """
    End-to-end NLP pipeline:
      1. Text cleaning
      2. N-gram extraction
      3. TF-IDF computation
      4. Topic modeling
      5. Sentiment scoring
    """

    def __init__(
        self,
        n_topics: int = 5,
        max_features: int = 2000,
        topic_method: str = "lda",
    ):
        self.preprocessor = TextPreprocessor()
        self.ngram_extractor = NGramExtractor(n=2, top_k=20)
        self.tfidf_analyzer = TFIDFAnalyzer(max_features=max_features)
        self.topic_modeler = TopicModeler(
            n_topics=n_topics, method=topic_method, max_features=max_features
        )
        self.sentiment_scorer = LexiconSentimentScorer()

    def run(self, corpus: Union[pd.Series, List[str]]) -> Dict[str, Any]:
        """
        Execute all pipeline steps.

        Returns
        -------
        dict with: 'clean_corpus', 'bigrams', 'top_terms', 'topics',
                   'dominant_topics', 'sentiment'
        """
        corpus = pd.Series(corpus)
        results = {}

        # 1. Clean
        clean = self.preprocessor.transform(corpus)
        results["clean_corpus"] = clean

        # 2. N-grams
        results["bigrams"] = self.ngram_extractor.extract(clean)

        # 3. TF-IDF
        self.tfidf_analyzer.fit_transform(clean)
        results["top_terms"] = self.tfidf_analyzer.global_top_terms(n=20)

        # 4. Topics
        self.topic_modeler.fit(clean)
        results["topics"] = self.topic_modeler.get_topics()
        results["dominant_topics"] = self.topic_modeler.dominant_topic(clean)

        # 5. Sentiment
        results["sentiment"] = self.sentiment_scorer.score_corpus(corpus)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "This product is amazing! Great quality and fast delivery. I love it.",
        "Terrible experience. The item was broken and customer service was awful.",
        "Decent product, nothing special. Works as expected.",
        "Absolutely fantastic! Best purchase I've made this year. Highly recommend.",
        "Very slow and unreliable. The app crashes constantly. Frustrating!",
        "Good value for money. Simple and intuitive interface.",
        "Do not buy this. Worst quality ever. Complete waste of money.",
        "Pretty good overall. A few minor issues but nothing major.",
    ] * 10  # inflate corpus for topic modeling

    pipeline = NLPPipeline(n_topics=3, max_features=500)
    results = pipeline.run(sample_texts)

    print("=== Top Bigrams ===")
    print(results["bigrams"].head())

    print("\n=== Top TF-IDF Terms ===")
    print(results["top_terms"].head(10))

    print("\n=== Topics ===")
    print(results["topics"])

    print("\n=== Sentiment Distribution ===")
    print(results["sentiment"]["label"].value_counts())
