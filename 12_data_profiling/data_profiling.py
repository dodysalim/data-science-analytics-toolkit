"""
Data Profiling & Quality Report Generator
==========================================
Automated dataset profiling that generates comprehensive quality reports:
- Column-level statistics (numeric, categorical, datetime)
- Missing value analysis
- Duplicate detection
- Cardinality and uniqueness analysis
- Distribution shape classification
- Data type inference and casting recommendations
- Overall quality score

Author: Data Science Analytics Toolkit
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# COLUMN PROFILERS
# ──────────────────────────────────────────────────────────────────────────────

class NumericProfiler:
    """Generate statistical profile for a numeric column."""

    def profile(self, series: pd.Series) -> Dict[str, Any]:
        """
        Compute descriptive statistics for a numeric series.

        Returns
        -------
        dict with: count, missing, missing_pct, mean, median, std, min, max,
                   q1, q3, iqr, skewness, kurtosis, zeros, zeros_pct,
                   negatives, is_integer, distribution_shape
        """
        clean = series.dropna()
        n_total = len(series)
        n_missing = series.isna().sum()

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        skew = float(clean.skew()) if len(clean) > 3 else np.nan
        kurt = float(clean.kurt()) if len(clean) > 3 else np.nan

        # Distribution shape heuristic
        if abs(skew) < 0.5:
            shape = "approximately_normal"
        elif skew > 1.5:
            shape = "highly_right_skewed"
        elif skew > 0.5:
            shape = "right_skewed"
        elif skew < -1.5:
            shape = "highly_left_skewed"
        else:
            shape = "left_skewed"

        return {
            "dtype": str(series.dtype),
            "count": int(len(clean)),
            "missing": int(n_missing),
            "missing_pct": round(n_missing / n_total * 100, 2),
            "mean": round(float(clean.mean()), 6),
            "median": round(float(clean.median()), 6),
            "mode": round(float(clean.mode().iloc[0]), 6) if len(clean) > 0 else None,
            "std": round(float(clean.std()), 6),
            "variance": round(float(clean.var()), 6),
            "min": round(float(clean.min()), 6),
            "max": round(float(clean.max()), 6),
            "range": round(float(clean.max() - clean.min()), 6),
            "q1": round(q1, 6),
            "q3": round(q3, 6),
            "iqr": round(q3 - q1, 6),
            "p5": round(float(clean.quantile(0.05)), 6),
            "p95": round(float(clean.quantile(0.95)), 6),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "zeros": int((clean == 0).sum()),
            "zeros_pct": round((clean == 0).sum() / n_total * 100, 2),
            "negatives": int((clean < 0).sum()),
            "is_integer_valued": bool(clean.dropna().apply(lambda x: x == int(x)).all()),
            "distribution_shape": shape,
            "outliers_iqr": int(((clean < q1 - 1.5 * (q3 - q1)) | (clean > q3 + 1.5 * (q3 - q1))).sum()),
        }


class CategoricalProfiler:
    """Generate statistical profile for a categorical/object column."""

    def profile(self, series: pd.Series, top_n: int = 10) -> Dict[str, Any]:
        """
        Compute statistics for a categorical series.

        Returns
        -------
        dict with: count, missing, missing_pct, n_unique, cardinality_ratio,
                   top_values, top_frequencies, is_binary, is_identifier
        """
        n_total = len(series)
        n_missing = series.isna().sum()
        clean = series.dropna()
        n_unique = int(clean.nunique())
        value_counts = clean.value_counts()
        top = value_counts.head(top_n)

        return {
            "dtype": str(series.dtype),
            "count": int(len(clean)),
            "missing": int(n_missing),
            "missing_pct": round(n_missing / n_total * 100, 2),
            "n_unique": n_unique,
            "cardinality_ratio": round(n_unique / len(clean) * 100, 2) if len(clean) > 0 else 0,
            "top_values": top.index.tolist(),
            "top_frequencies": top.values.tolist(),
            "top_proportions": (top / len(clean)).round(4).values.tolist(),
            "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "most_frequent_pct": round(value_counts.iloc[0] / n_total * 100, 2) if len(value_counts) > 0 else 0,
            "is_binary": n_unique == 2,
            "is_potential_identifier": n_unique / len(clean) > 0.95 if len(clean) > 0 else False,
            "is_constant": n_unique == 1,
            "entropy": round(float(stats.entropy(value_counts / len(clean), base=2)), 4) if len(value_counts) > 0 else 0,
        }


class DatetimeProfiler:
    """Generate statistical profile for a datetime column."""

    def profile(self, series: pd.Series) -> Dict[str, Any]:
        """Compute temporal statistics."""
        try:
            dt = pd.to_datetime(series, errors="coerce")
        except Exception:
            return {"error": "Could not parse as datetime"}

        n_total = len(dt)
        n_missing = dt.isna().sum()
        clean = dt.dropna()

        return {
            "dtype": str(series.dtype),
            "count": int(len(clean)),
            "missing": int(n_missing),
            "missing_pct": round(n_missing / n_total * 100, 2),
            "min_date": str(clean.min()),
            "max_date": str(clean.max()),
            "date_range_days": int((clean.max() - clean.min()).days) if len(clean) > 1 else 0,
            "n_unique_dates": int(clean.dt.date.nunique()),
            "most_common_year": int(clean.dt.year.mode().iloc[0]) if len(clean) > 0 else None,
            "most_common_month": int(clean.dt.month.mode().iloc[0]) if len(clean) > 0 else None,
            "most_common_day_of_week": int(clean.dt.dayofweek.mode().iloc[0]) if len(clean) > 0 else None,
            "has_time_component": bool((clean.dt.hour != 0).any()),
        }


# ──────────────────────────────────────────────────────────────────────────────
# DATASET PROFILER
# ──────────────────────────────────────────────────────────────────────────────

class DatasetProfiler:
    """
    Orchestrate full dataset profiling across all column types.

    Returns a structured profile report with:
    - Per-column statistics
    - Missing value heatmap data
    - Duplicate analysis
    - Overall data quality score
    """

    def __init__(self, datetime_cols: Optional[List[str]] = None):
        self.datetime_cols = datetime_cols or []
        self._numeric_profiler = NumericProfiler()
        self._cat_profiler = CategoricalProfiler()
        self._dt_profiler = DatetimeProfiler()

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full profiling on a DataFrame.

        Returns
        -------
        dict with: 'overview', 'columns', 'missing', 'duplicates', 'quality_score'
        """
        report = {}

        # ── Overview ──────────────────────────────────────────────────────────
        report["overview"] = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_cells": len(df) * len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
            "n_numeric_cols": len(df.select_dtypes(include=np.number).columns),
            "n_categorical_cols": len(df.select_dtypes(include=["object", "category"]).columns),
            "n_datetime_cols": len(df.select_dtypes(include=["datetime64"]).columns),
            "total_missing": int(df.isna().sum().sum()),
            "total_missing_pct": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        }

        # ── Column profiles ───────────────────────────────────────────────────
        columns = {}
        for col in df.columns:
            if col in self.datetime_cols or pd.api.types.is_datetime64_any_dtype(df[col]):
                columns[col] = {"type": "datetime", **self._dt_profiler.profile(df[col])}
            elif pd.api.types.is_numeric_dtype(df[col]):
                columns[col] = {"type": "numeric", **self._numeric_profiler.profile(df[col])}
            else:
                columns[col] = {"type": "categorical", **self._cat_profiler.profile(df[col])}
        report["columns"] = columns

        # ── Missing analysis ──────────────────────────────────────────────────
        missing_series = df.isna().sum()
        missing_pct = (df.isna().sum() / len(df) * 100).round(2)
        report["missing"] = pd.DataFrame({
            "missing_count": missing_series,
            "missing_pct": missing_pct,
        }).sort_values("missing_pct", ascending=False).to_dict()

        # ── Duplicates ────────────────────────────────────────────────────────
        n_dupes = int(df.duplicated().sum())
        report["duplicates"] = {
            "n_duplicate_rows": n_dupes,
            "duplicate_pct": round(n_dupes / len(df) * 100, 2),
            "is_concern": n_dupes > 0,
        }

        # ── Quality Score ─────────────────────────────────────────────────────
        report["quality_score"] = self._compute_quality_score(df, report)

        return report

    def _compute_quality_score(self, df: pd.DataFrame, report: Dict) -> Dict[str, Any]:
        """
        Compute a 0-100 data quality score based on:
        - Completeness (missing values penalty)
        - Uniqueness (duplicate rows penalty)
        - Consistency (constant column penalty)
        - Validity (potential identifier columns flagged as categorical)
        """
        completeness = 100 - report["overview"]["total_missing_pct"]
        uniqueness = 100 - report["duplicates"]["duplicate_pct"]

        n_constant = sum(
            1 for v in report["columns"].values()
            if v.get("is_constant", False)
        )
        n_identifiers = sum(
            1 for v in report["columns"].values()
            if v.get("is_potential_identifier", False)
        )
        consistency = max(0, 100 - n_constant * 10)
        validity = max(0, 100 - n_identifiers * 5)

        overall = round((completeness + uniqueness + consistency + validity) / 4, 1)
        grade = "A" if overall >= 90 else "B" if overall >= 75 else "C" if overall >= 60 else "D"

        return {
            "completeness": round(completeness, 2),
            "uniqueness": round(uniqueness, 2),
            "consistency": round(consistency, 2),
            "validity": round(validity, 2),
            "overall": overall,
            "grade": grade,
        }

    def summary_table(self, report: Dict) -> pd.DataFrame:
        """Return a tidy summary table of all column profiles."""
        rows = []
        for col, prof in report["columns"].items():
            row = {"column": col, "type": prof.get("type")}
            row["missing_pct"] = prof.get("missing_pct")
            row["n_unique"] = prof.get("n_unique") or prof.get("n_unique_dates")
            if prof["type"] == "numeric":
                row["mean"] = prof.get("mean")
                row["std"] = prof.get("std")
                row["min"] = prof.get("min")
                row["max"] = prof.get("max")
                row["skewness"] = prof.get("skewness")
                row["outliers"] = prof.get("outliers_iqr")
                row["distribution"] = prof.get("distribution_shape")
            elif prof["type"] == "categorical":
                row["most_frequent"] = prof.get("most_frequent")
                row["most_frequent_pct"] = prof.get("most_frequent_pct")
                row["cardinality_ratio"] = prof.get("cardinality_ratio")
            rows.append(row)
        return pd.DataFrame(rows).set_index("column")

    def export_json(self, report: Dict, path: str) -> None:
        """Export the full report to a JSON file."""
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=convert)

    def print_summary(self, report: Dict) -> None:
        """Print a formatted summary to stdout."""
        ov = report["overview"]
        qs = report["quality_score"]
        print("=" * 60)
        print("  DATA PROFILING REPORT")
        print("=" * 60)
        print(f"  Rows:              {ov['n_rows']:,}")
        print(f"  Columns:           {ov['n_cols']}")
        print(f"  Memory:            {ov['memory_mb']} MB")
        print(f"  Total Missing:     {ov['total_missing']:,} ({ov['total_missing_pct']}%)")
        print(f"  Duplicate Rows:    {report['duplicates']['n_duplicate_rows']:,}")
        print("-" * 60)
        print("  DATA QUALITY SCORE")
        print(f"  Completeness:  {qs['completeness']}%")
        print(f"  Uniqueness:    {qs['uniqueness']}%")
        print(f"  Consistency:   {qs['consistency']}%")
        print(f"  Overall:       {qs['overall']} / 100  [Grade: {qs['grade']}]")
        print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "age": np.random.randint(18, 80, n).astype(float),
        "income": np.random.lognormal(10, 0.8, n),
        "score": np.random.beta(2, 5, n) * 100,
        "segment": np.random.choice(["Premium", "Standard", "Basic", "Trial"], n),
        "country": np.random.choice(["US", "UK", "DE", "FR", "BR"], n),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="H"),
        "is_active": np.random.choice([True, False], n),
    })

    # Introduce 5% missing values
    for col in ["age", "income", "score", "country"]:
        df.loc[df.sample(frac=0.05).index, col] = np.nan

    # Add a few duplicates
    df = pd.concat([df, df.sample(20)], ignore_index=True)

    profiler = DatasetProfiler()
    report = profiler.profile(df)
    profiler.print_summary(report)

    print("\n=== Column Summary Table ===")
    print(profiler.summary_table(report).to_string())
