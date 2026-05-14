# 13 — A/B Testing & Experiment Analysis Toolkit

Full statistical framework for designing, running, and reporting controlled experiments — both frequentist and Bayesian approaches.

## Classes

| Class | Description |
|---|---|
| `SampleSizeCalculator` | Power analysis for proportions and means, MDE computation |
| `FrequentistABTester` | Z-test, Welch's t-test, Mann-Whitney U, chi-square |
| `BayesianABTester` | Beta-Binomial conjugate model, P(treatment > control), expected lift |
| `ExperimentReporter` | Markdown A/B test report generation |

## Workflow

```python
from ab_testing import SampleSizeCalculator, FrequentistABTester, BayesianABTester, ExperimentReporter

# 1. Plan sample size
calc = SampleSizeCalculator()
plan = calc.for_proportions(baseline_rate=0.05, min_detectable_effect=0.20)
print(f"Need {plan['n_per_group']:,} users per group")

# 2. Run frequentist test
tester = FrequentistABTester()
result = tester.proportion_test(
    control_conversions=500, control_n=10000,
    treatment_conversions=580, treatment_n=10000,
)
print(result["recommendation"])

# 3. Bayesian analysis
bay = BayesianABTester()
bay_result = bay.analyze(500, 10000, 580, 10000)
print(f"P(treatment wins): {bay_result['prob_treatment_wins']:.1%}")

# 4. Generate report
reporter = ExperimentReporter()
md = reporter.generate("Homepage CTA Test", result, bay_result, plan)
```

## Requirements

```
scipy>=1.10.0
numpy>=1.23.0
pandas>=1.5.0
```
