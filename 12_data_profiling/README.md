# 12 — Data Profiling & Quality Report Generator

Automated dataset profiling with comprehensive per-column statistics, missing value analysis, duplicate detection, and a 0-100 data quality score with letter grade.

## Classes

| Class | Description |
|---|---|
| `NumericProfiler` | Mean, median, std, IQR, skewness, kurtosis, outlier count, distribution shape |
| `CategoricalProfiler` | Cardinality, top values, entropy, binary/identifier detection |
| `DatetimeProfiler` | Date range, temporal patterns, time component detection |
| `DatasetProfiler` | Full orchestration: overview, column profiles, missing analysis, quality score |

## Quality Score

The overall quality score (0–100) and letter grade (A/B/C/D) is computed from 4 dimensions:

| Dimension | Description |
|---|---|
| Completeness | Penalizes missing values |
| Uniqueness | Penalizes duplicate rows |
| Consistency | Penalizes constant/useless columns |
| Validity | Flags potential identifier columns stored as categoricals |

## Usage

```python
from data_profiling import DatasetProfiler

profiler = DatasetProfiler()
report = profiler.profile(df)

# Print formatted summary
profiler.print_summary(report)

# Get tidy table
print(profiler.summary_table(report))

# Export to JSON
profiler.export_json(report, "profile_report.json")
```

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0
```
