"""Pruebas de los módulos reutilizables (datos sintéticos, sin el CSV grande)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MOD = Path(__file__).resolve().parents[1] / "src" / "modules"


def load(folder, name):
    spec = importlib.util.spec_from_file_location(name, MOD / folder / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cleaning = load("02_data_cleaning_utils", "data_cleaning")
stats = load("05_statistical_analysis", "statistical_analysis")
ab = load("13_ab_testing", "ab_testing")
profiling = load("12_data_profiling", "data_profiling")


@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1.0, 2.0, None, 4.0, 4.0, 100.0],
        "b": ["x", "y", "y", None, None, "x"],
        "c": [10, 20, 30, 40, 40, 50],
    })


def test_report_nulls_counts_missing(df):
    rep = cleaning.report_nulls(df)
    assert rep.shape[0] >= 1


def test_fill_nulls_leaves_no_missing(df):
    out = cleaning.fill_nulls_smart(df)
    assert out.isna().sum().sum() == 0
    assert len(out) == len(df)


def test_remove_duplicates(df):
    out = cleaning.remove_duplicates(df)
    assert len(out) == len(df.drop_duplicates())


def test_iqr_removes_extreme_value():
    d = pd.DataFrame({"v": [1, 2, 3, 2, 3, 2, 1000]})
    out = cleaning.remove_outliers_iqr(d, cols=["v"])
    assert 1000 not in out["v"].values


def test_minmax_normalization_range():
    d = pd.DataFrame({"v": [5.0, 10.0, 15.0]})
    out = cleaning.normalize_columns(d, cols=["v"], method="minmax")
    assert out["v"].min() == pytest.approx(0.0) and out["v"].max() == pytest.approx(1.0)


def test_confidence_interval_contains_mean():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(50, 5, 500))
    ci = stats.confidence_interval(s)
    lo, hi = ci[0], ci[1]
    assert lo < s.mean() < hi


def test_sample_size_is_positive():
    res = ab.SampleSizeCalculator().for_proportions(0.10, 0.02)
    nums = [v for v in res.values() if isinstance(v, (int, float, np.integer, np.floating))]
    assert nums and max(nums) > 0


def test_proportion_test_detects_clear_difference():
    res = ab.FrequentistABTester().proportion_test(100, 1000, 200, 1000)
    p = next(v for k, v in res.items() if "p" in k.lower() and "valor" in k.lower() or k.lower() in ("p_value", "pvalue", "p"))
    assert p < 0.05


def test_dataset_profiler_runs(df):
    rep = profiling.DatasetProfiler().profile(df)
    assert isinstance(rep, dict) and rep
