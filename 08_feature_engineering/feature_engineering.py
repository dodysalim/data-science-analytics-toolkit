"""
Feature Engineering Toolkit
============================
SOLID-compliant library for automated and manual feature engineering:
- Date/time feature extraction
- Polynomial and interaction features
- Target encoding & frequency encoding
- Binning and discretization
- Feature selection via statistical filters
- Full pipeline orchestration

Author: Data Science Analytics Toolkit
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, chi2
)


# ──────────────────────────────────────────────────────────────────────────────
# DATE / TIME FEATURES
# ──────────────────────────────────────────────────────────────────────────────

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract rich temporal features from datetime columns.

    Features extracted (configurable):
        year, month, day, hour, minute, day_of_week, day_of_year,
        week_of_year, quarter, is_weekend, is_month_start, is_month_end,
        sin/cos cyclical encodings for month, day_of_week, and hour.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        cyclical_encoding: bool = True,
        drop_original: bool = False,
    ):
        self.columns = columns
        self.cyclical_encoding = cyclical_encoding
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None) -> "DateTimeFeatureExtractor":
        """Detect datetime columns if none specified."""
        if self.columns is None:
            self._cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
        else:
            self._cols = self.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features from all datetime columns."""
        df = X.copy()
        for col in self._cols:
            dt = pd.to_datetime(df[col])
            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month
            df[f"{col}_day"] = dt.dt.day
            df[f"{col}_hour"] = dt.dt.hour
            df[f"{col}_day_of_week"] = dt.dt.dayofweek
            df[f"{col}_day_of_year"] = dt.dt.dayofyear
            df[f"{col}_week_of_year"] = dt.dt.isocalendar().week.astype(int)
            df[f"{col}_quarter"] = dt.dt.quarter
            df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
            df[f"{col}_is_month_start"] = dt.dt.is_month_start.astype(int)
            df[f"{col}_is_month_end"] = dt.dt.is_month_end.astype(int)

            if self.cyclical_encoding:
                df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
                df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
                df[f"{col}_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
                df[f"{col}_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
                df[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
                df[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)

            if self.drop_original:
                df = df.drop(columns=[col])

        return df


# ──────────────────────────────────────────────────────────────────────────────
# POLYNOMIAL & INTERACTION FEATURES
# ──────────────────────────────────────────────────────────────────────────────

class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate polynomial and interaction features for numeric columns.

    Wraps sklearn PolynomialFeatures with a pandas-friendly interface.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
    ):
        self.columns = columns
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self._poly = None
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "PolynomialFeatureGenerator":
        cols = self.columns or X.select_dtypes(include=np.number).columns.tolist()
        self._cols = cols
        self._poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        self._poly.fit(X[cols])
        self._feature_names = self._poly.get_feature_names_out(cols).tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        poly_array = self._poly.transform(X[self._cols])
        poly_df = pd.DataFrame(poly_array, columns=self._feature_names, index=X.index)
        # Drop original columns to avoid duplication, keep new ones
        original_cols = [c for c in X.columns if c not in self._cols]
        return pd.concat([X[original_cols].reset_index(drop=True),
                          poly_df.reset_index(drop=True)], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# TARGET ENCODING
# ──────────────────────────────────────────────────────────────────────────────

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Mean target encoding for categorical variables with smoothing
    to prevent overfitting on rare categories.

    Formula (smoothed):
        encoded = (n * category_mean + m * global_mean) / (n + m)
    where m is the smoothing factor (min_samples_leaf).
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        smoothing: float = 10.0,
    ):
        self.columns = columns
        self.smoothing = smoothing
        self._encodings: Dict[str, Dict[Any, float]] = {}
        self._global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        cols = self.columns or X.select_dtypes(include=["object", "category"]).columns.tolist()
        self._cols = cols
        self._global_mean = float(y.mean())
        for col in cols:
            df_tmp = pd.DataFrame({"cat": X[col], "target": y})
            stats = df_tmp.groupby("cat")["target"].agg(["count", "mean"])
            smooth = (
                (stats["count"] * stats["mean"] + self.smoothing * self._global_mean)
                / (stats["count"] + self.smoothing)
            )
            self._encodings[col] = smooth.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self._cols:
            df[f"{col}_target_enc"] = df[col].map(self._encodings[col]).fillna(self._global_mean)
        return df


# ──────────────────────────────────────────────────────────────────────────────
# FREQUENCY ENCODING
# ──────────────────────────────────────────────────────────────────────────────

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables by their frequency (proportion) in the training set.
    Useful when cardinality is high and order doesn't matter.
    """

    def __init__(self, columns: Optional[List[str]] = None, normalize: bool = True):
        self.columns = columns
        self.normalize = normalize
        self._freq_maps: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        cols = self.columns or X.select_dtypes(include=["object", "category"]).columns.tolist()
        self._cols = cols
        for col in cols:
            freq = X[col].value_counts(normalize=self.normalize)
            self._freq_maps[col] = freq.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self._cols:
            df[f"{col}_freq_enc"] = df[col].map(self._freq_maps[col]).fillna(0)
        return df


# ──────────────────────────────────────────────────────────────────────────────
# BINNING / DISCRETIZATION
# ──────────────────────────────────────────────────────────────────────────────

class Binner(BaseEstimator, TransformerMixin):
    """
    Discretize continuous numerical features into bins.
    Supports equal-width, equal-frequency (quantile), and custom bins.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        n_bins: int = 5,
        strategy: str = "quantile",
        labels: Optional[List[str]] = None,
    ):
        self.columns = columns
        self.n_bins = n_bins
        self.strategy = strategy
        self.labels = labels
        self._bin_edges: Dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "Binner":
        cols = self.columns or X.select_dtypes(include=np.number).columns.tolist()
        self._cols = cols
        for col in cols:
            if self.strategy == "quantile":
                quantiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.percentile(X[col].dropna(), quantiles)
            elif self.strategy == "uniform":
                edges = np.linspace(X[col].min(), X[col].max(), self.n_bins + 1)
            else:
                raise ValueError("strategy must be 'quantile' or 'uniform'.")
            self._bin_edges[col] = np.unique(edges)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self._cols:
            df[f"{col}_binned"] = pd.cut(
                df[col],
                bins=self._bin_edges[col],
                labels=self.labels,
                include_lowest=True,
                duplicates="drop",
            )
        return df


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ──────────────────────────────────────────────────────────────────────────────

class StatisticalFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select top-k features using statistical scoring functions.

    Supports:
        - f_classif       : ANOVA F-test (classification)
        - f_regression    : F-test (regression)
        - mutual_info     : Mutual information (classification)
    """

    SCORE_FUNCS = {
        "f_classif": f_classif,
        "f_regression": f_regression,
        "mutual_info": mutual_info_classif,
    }

    def __init__(
        self,
        score_func: str = "f_classif",
        k: Union[int, str] = 10,
    ):
        if score_func not in self.SCORE_FUNCS:
            raise ValueError(f"score_func must be one of {list(self.SCORE_FUNCS.keys())}")
        self.score_func = score_func
        self.k = k
        self._selector = None
        self._selected_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StatisticalFeatureSelector":
        numeric_X = X.select_dtypes(include=np.number)
        self._all_numeric_cols = numeric_X.columns.tolist()
        self._selector = SelectKBest(
            score_func=self.SCORE_FUNCS[self.score_func],
            k=min(self.k, len(self._all_numeric_cols)) if isinstance(self.k, int) else self.k,
        )
        self._selector.fit(numeric_X.fillna(0), y)
        mask = self._selector.get_support()
        self._selected_columns = [c for c, s in zip(self._all_numeric_cols, mask) if s]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        non_numeric = X.select_dtypes(exclude=np.number)
        return pd.concat([non_numeric, X[self._selected_columns]], axis=1)

    def feature_scores(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        scores = self._selector.scores_
        return (
            pd.DataFrame({"feature": self._all_numeric_cols, "score": scores})
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )


# ──────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

class FeatureEngineeringPipeline:
    """
    Orchestrate all feature engineering steps:
      1. DateTime extraction
      2. Frequency encoding for categoricals
      3. Target encoding (if target provided)
      4. Polynomial features
      5. Binning
      6. Feature selection
    """

    def __init__(
        self,
        datetime_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        poly_degree: int = 2,
        n_bins: int = 5,
        k_best: int = 20,
        task: str = "classification",
    ):
        self.datetime_cols = datetime_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.poly_degree = poly_degree
        self.n_bins = n_bins
        self.k_best = k_best
        self.task = task

        self._steps: List[Any] = []

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform the dataset through all feature engineering steps."""
        df = X.copy()

        # 1. DateTime
        if self.datetime_cols:
            dt_extractor = DateTimeFeatureExtractor(columns=self.datetime_cols, drop_original=True)
            df = dt_extractor.fit_transform(df)
            self._steps.append(("datetime", dt_extractor))

        # 2. Frequency encoding
        freq_encoder = FrequencyEncoder(columns=self.cat_cols)
        df = freq_encoder.fit_transform(df)
        self._steps.append(("freq_enc", freq_encoder))

        # 3. Target encoding (only if y provided)
        if y is not None and self.cat_cols:
            target_encoder = TargetEncoder(columns=self.cat_cols)
            df = target_encoder.fit(df, y).transform(df)
            self._steps.append(("target_enc", target_encoder))

        # 4. Binning
        binner = Binner(columns=self.num_cols, n_bins=self.n_bins)
        df = binner.fit_transform(df)
        self._steps.append(("binner", binner))

        # 5. Feature selection
        if y is not None:
            score_func = "f_classif" if self.task == "classification" else "f_regression"
            selector = StatisticalFeatureSelector(score_func=score_func, k=self.k_best)
            df = selector.fit_transform(df, y)
            self._steps.append(("selector", selector))

        return df

    def get_feature_scores(self) -> Optional[pd.DataFrame]:
        """Retrieve feature scores from the selector step (if applied)."""
        for name, step in self._steps:
            if name == "selector":
                return step.feature_scores()
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    df = pd.DataFrame(
        {
            "signup_date": pd.date_range("2022-01-01", periods=n, freq="D"),
            "category": np.random.choice(["A", "B", "C", "D"], n),
            "region": np.random.choice(["North", "South", "East", "West"], n),
            "age": np.random.randint(18, 70, n),
            "income": np.random.normal(50000, 15000, n),
            "score": np.random.uniform(0, 100, n),
        }
    )
    y = pd.Series(np.random.randint(0, 2, n), name="churn")

    pipeline = FeatureEngineeringPipeline(
        datetime_cols=["signup_date"],
        cat_cols=["category", "region"],
        num_cols=["age", "income", "score"],
        poly_degree=2,
        n_bins=5,
        k_best=15,
        task="classification",
    )
    df_engineered = pipeline.fit_transform(df, y)
    print(f"Original shape: {df.shape}")
    print(f"Engineered shape: {df_engineered.shape}")
    print("\nTop features by F-score:")
    print(pipeline.get_feature_scores().head(10))
