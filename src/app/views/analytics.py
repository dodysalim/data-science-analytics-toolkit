"""Cálculos de las vistas (funciones puras, sin Streamlit)."""
from typing import Optional, Sequence

import pandas as pd


def share_table(df: pd.DataFrame, index: str, column: str, order: Optional[Sequence[str]] = None,
                columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Proporción de cada valor de `column` dentro de cada grupo de `index` (filas suman 1)."""
    counts = df.groupby([index, column]).size().unstack(fill_value=0)
    if order is not None:
        counts = counts.reindex([v for v in order if v in counts.index])
    shares = counts.div(counts.sum(axis=1), axis=0)
    if columns is not None:
        shares = shares.reindex(columns=[c for c in columns if c in shares.columns])
    return shares


def rate_by(df: pd.DataFrame, group: str, column: str, value: str) -> pd.Series:
    """% de filas con `column == value` en cada grupo, de mayor a menor."""
    return (df[column].eq(value).groupby(df[group]).mean() * 100).sort_values(ascending=False)


def monthly_rate(df: pd.DataFrame, column: str, value: str, month_col: str = "mes") -> pd.Series:
    return (df[column].eq(value).groupby(df[month_col]).mean() * 100).fillna(0)


def resolution_counts(df: pd.DataFrame, channel: str, outcome: str = "Resolved"):
    subset = df[df["channel"] == channel]
    return int((subset["outcome"] == outcome).sum()), len(subset)
