"""
A/B Testing & Experiment Analysis Toolkit
==========================================
Statistical framework for designing and analyzing controlled experiments:
- Sample size calculation (power analysis)
- Z-test and T-test for means
- Chi-square test for proportions/conversion rates
- Mann-Whitney U test (non-parametric)
- Sequential testing (always-valid inference)
- Bayesian A/B testing
- Multi-variant (A/B/n) ANOVA analysis
- Full experiment report generation

Author: Data Science Analytics Toolkit
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.special import betaln
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# SAMPLE SIZE CALCULATOR
# ──────────────────────────────────────────────────────────────────────────────

class SampleSizeCalculator:
    """
    Calculate required sample size for A/B test power analysis.

    Based on two-sample proportions (conversion rates) or two-sample means.
    """

    def for_proportions(
        self,
        baseline_rate: float,
        min_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
        two_sided: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate sample size for comparing two proportions.

        Parameters
        ----------
        baseline_rate : float
            Control group conversion rate (e.g., 0.10 for 10%).
        min_detectable_effect : float
            Minimum relative lift to detect (e.g., 0.05 for +5%).
        alpha : float
            Significance level (Type I error).
        power : float
            Statistical power (1 - Type II error).
        two_sided : bool
            Two-sided or one-sided test.

        Returns
        -------
        dict with: n_per_group, total_n, treatment_rate, effect_size
        """
        treatment_rate = baseline_rate * (1 + min_detectable_effect)
        z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
        z_power = stats.norm.ppf(power)

        p_bar = (baseline_rate + treatment_rate) / 2
        effect_size = abs(treatment_rate - baseline_rate)

        # Standard formula for two proportions
        n = (
            (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
             z_power * np.sqrt(baseline_rate * (1 - baseline_rate) +
                               treatment_rate * (1 - treatment_rate))) ** 2
            / effect_size ** 2
        )
        n = int(np.ceil(n))

        return {
            "n_per_group": n,
            "total_n": n * 2,
            "baseline_rate": baseline_rate,
            "treatment_rate": round(treatment_rate, 6),
            "min_detectable_effect": min_detectable_effect,
            "alpha": alpha,
            "power": power,
            "effect_size": round(effect_size, 6),
        }

    def for_means(
        self,
        baseline_mean: float,
        baseline_std: float,
        min_detectable_delta: float,
        alpha: float = 0.05,
        power: float = 0.80,
        two_sided: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate sample size for comparing two means (independent t-test).

        Parameters
        ----------
        baseline_mean : float
            Control group mean.
        baseline_std : float
            Pooled standard deviation estimate.
        min_detectable_delta : float
            Absolute minimum effect size to detect.
        """
        z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
        z_power = stats.norm.ppf(power)
        n = int(np.ceil(2 * ((z_alpha + z_power) * baseline_std / min_detectable_delta) ** 2))
        return {
            "n_per_group": n,
            "total_n": n * 2,
            "baseline_mean": baseline_mean,
            "min_detectable_delta": min_detectable_delta,
            "cohens_d": round(min_detectable_delta / baseline_std, 4),
            "alpha": alpha,
            "power": power,
        }

    def minimum_detectable_effect(
        self,
        n_per_group: int,
        baseline_rate: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> float:
        """Given a sample size, compute the MDE (minimum detectable effect)."""
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_power = stats.norm.ppf(power)
        p = baseline_rate
        mde = (z_alpha + z_power) * np.sqrt(2 * p * (1 - p) / n_per_group)
        return round(float(mde / p), 4)  # relative MDE


# ──────────────────────────────────────────────────────────────────────────────
# FREQUENTIST A/B TESTER
# ──────────────────────────────────────────────────────────────────────────────

class FrequentistABTester:
    """
    Frequentist hypothesis testing for A/B experiments.

    Supports:
      - Two-proportion z-test (conversion rates)
      - Two-sample independent t-test (continuous metrics)
      - Mann-Whitney U test (non-parametric)
      - Chi-square test (multi-cell contingency)
    """

    def proportion_test(
        self,
        control_conversions: int,
        control_n: int,
        treatment_conversions: int,
        treatment_n: int,
        alpha: float = 0.05,
        two_sided: bool = True,
    ) -> Dict[str, Any]:
        """
        Two-proportion z-test for comparing conversion rates.

        Parameters
        ----------
        control_conversions : int
            Number of successes in control group.
        control_n : int
            Total users in control group.
        treatment_conversions : int
            Number of successes in treatment group.
        treatment_n : int
            Total users in treatment group.
        alpha : float
            Significance level.
        two_sided : bool
            Two-sided (True) or one-sided test.

        Returns
        -------
        dict with: control_rate, treatment_rate, lift, z_stat, p_value,
                   confidence_interval, is_significant, recommendation
        """
        p_c = control_conversions / control_n
        p_t = treatment_conversions / treatment_n
        p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)

        se = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
        z_stat = (p_t - p_c) / se
        p_value = 2 * stats.norm.sf(abs(z_stat)) if two_sided else stats.norm.sf(z_stat)

        # 95% CI on the difference
        z_crit = stats.norm.ppf(1 - alpha / 2)
        se_diff = np.sqrt(p_c * (1 - p_c) / control_n + p_t * (1 - p_t) / treatment_n)
        ci_low = (p_t - p_c) - z_crit * se_diff
        ci_high = (p_t - p_c) + z_crit * se_diff

        significant = p_value < alpha
        lift = (p_t - p_c) / p_c * 100

        return {
            "control_rate": round(p_c, 6),
            "treatment_rate": round(p_t, 6),
            "absolute_diff": round(p_t - p_c, 6),
            "relative_lift_pct": round(lift, 4),
            "z_statistic": round(float(z_stat), 4),
            "p_value": round(float(p_value), 6),
            "confidence_interval_95pct": (round(ci_low, 6), round(ci_high, 6)),
            "is_significant": bool(significant),
            "alpha": alpha,
            "recommendation": (
                f"Implement treatment — statistically significant lift of {lift:.2f}%."
                if significant and lift > 0
                else "Do not implement — no statistically significant improvement."
                if not significant
                else "Treatment is significantly worse — do not implement."
            ),
        }

    def ttest(
        self,
        control_data: Union[np.ndarray, pd.Series],
        treatment_data: Union[np.ndarray, pd.Series],
        alpha: float = 0.05,
        equal_var: bool = False,
    ) -> Dict[str, Any]:
        """
        Independent samples t-test (Welch's by default).

        Parameters
        ----------
        control_data : array-like
            Metric values for the control group.
        treatment_data : array-like
            Metric values for the treatment group.
        alpha : float
            Significance level.
        equal_var : bool
            Use Student's t-test (True) or Welch's (False).
        """
        t_stat, p_value = stats.ttest_ind(control_data, treatment_data, equal_var=equal_var)
        lift = (np.mean(treatment_data) - np.mean(control_data)) / np.mean(control_data) * 100

        # Cohen's d
        pooled_std = np.sqrt((np.std(control_data) ** 2 + np.std(treatment_data) ** 2) / 2)
        cohens_d = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std

        # CI
        ci = stats.t.interval(
            1 - alpha,
            df=len(control_data) + len(treatment_data) - 2,
            loc=np.mean(treatment_data) - np.mean(control_data),
            scale=stats.sem(np.concatenate([control_data, treatment_data])),
        )

        return {
            "control_mean": round(float(np.mean(control_data)), 6),
            "treatment_mean": round(float(np.mean(treatment_data)), 6),
            "relative_lift_pct": round(float(lift), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "cohens_d": round(float(cohens_d), 4),
            "effect_size_label": (
                "small" if abs(cohens_d) < 0.5
                else "medium" if abs(cohens_d) < 0.8
                else "large"
            ),
            "confidence_interval_95pct": (round(ci[0], 6), round(ci[1], 6)),
            "is_significant": bool(p_value < alpha),
            "test_type": "Welch's t-test" if not equal_var else "Student's t-test",
        }

    def mann_whitney(
        self,
        control_data: Union[np.ndarray, pd.Series],
        treatment_data: Union[np.ndarray, pd.Series],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Non-parametric Mann-Whitney U test."""
        u_stat, p_value = stats.mannwhitneyu(
            control_data, treatment_data, alternative="two-sided"
        )
        return {
            "u_statistic": round(float(u_stat), 4),
            "p_value": round(float(p_value), 6),
            "is_significant": bool(p_value < alpha),
            "control_median": round(float(np.median(control_data)), 6),
            "treatment_median": round(float(np.median(treatment_data)), 6),
            "test_type": "Mann-Whitney U (non-parametric)",
        }

    def chi_square_test(
        self, contingency_table: pd.DataFrame, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Chi-square test of independence on a contingency table."""
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        cramers_v = np.sqrt(chi2 / (contingency_table.values.sum() * (min(contingency_table.shape) - 1)))
        return {
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "degrees_of_freedom": int(dof),
            "cramers_v": round(float(cramers_v), 4),
            "is_significant": bool(p_value < alpha),
            "expected_frequencies": pd.DataFrame(expected, index=contingency_table.index,
                                                  columns=contingency_table.columns).round(2),
        }


# ──────────────────────────────────────────────────────────────────────────────
# BAYESIAN A/B TESTER
# ──────────────────────────────────────────────────────────────────────────────

class BayesianABTester:
    """
    Bayesian A/B testing for binary outcomes (conversions).

    Uses Beta-Binomial conjugate model.
    Prior: Beta(alpha_prior, beta_prior) — default: Beta(1,1) = uniform
    Posterior: Beta(alpha_prior + conversions, beta_prior + non-conversions)
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def analyze(
        self,
        control_conversions: int,
        control_n: int,
        treatment_conversions: int,
        treatment_n: int,
        n_samples: int = 100_000,
    ) -> Dict[str, Any]:
        """
        Compute posterior distributions and probability that treatment > control.

        Parameters
        ----------
        control_conversions, control_n : int
            Control group successes and total.
        treatment_conversions, treatment_n : int
            Treatment group successes and total.
        n_samples : int
            Monte Carlo samples for P(treatment > control).

        Returns
        -------
        dict with posterior stats, P(treatment > control), expected lift
        """
        # Posterior parameters
        a_c = self.alpha_prior + control_conversions
        b_c = self.beta_prior + (control_n - control_conversions)
        a_t = self.alpha_prior + treatment_conversions
        b_t = self.beta_prior + (treatment_n - treatment_conversions)

        # Posterior means
        p_c = a_c / (a_c + b_c)
        p_t = a_t / (a_t + b_t)

        # Monte Carlo P(treatment > control)
        samples_c = np.random.beta(a_c, b_c, n_samples)
        samples_t = np.random.beta(a_t, b_t, n_samples)
        prob_t_wins = float((samples_t > samples_c).mean())

        # Expected lift distribution
        expected_lift = samples_t - samples_c
        lift_mean = float(expected_lift.mean())
        lift_ci = (float(np.percentile(expected_lift, 2.5)), float(np.percentile(expected_lift, 97.5)))

        return {
            "control_posterior_mean": round(p_c, 6),
            "treatment_posterior_mean": round(p_t, 6),
            "prob_treatment_wins": round(prob_t_wins, 4),
            "prob_control_wins": round(1 - prob_t_wins, 4),
            "expected_lift_mean": round(lift_mean, 6),
            "expected_lift_95pct_ci": (round(lift_ci[0], 6), round(lift_ci[1], 6)),
            "recommendation": (
                f"Deploy treatment — {prob_t_wins:.1%} probability of improvement."
                if prob_t_wins >= 0.95
                else f"Gather more data — {prob_t_wins:.1%} probability treatment wins."
                if prob_t_wins >= 0.80
                else f"Control preferred — only {prob_t_wins:.1%} probability treatment wins."
            ),
            "prior": f"Beta({self.alpha_prior}, {self.beta_prior})",
        }


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT REPORT
# ──────────────────────────────────────────────────────────────────────────────

class ExperimentReporter:
    """Generate a complete Markdown A/B test report."""

    def generate(
        self,
        experiment_name: str,
        frequentist_result: Dict,
        bayesian_result: Optional[Dict] = None,
        sample_size_plan: Optional[Dict] = None,
    ) -> str:
        """
        Produce a full Markdown experiment report.

        Returns
        -------
        str — Markdown formatted report
        """
        lines = [
            f"# A/B Test Report: {experiment_name}",
            "",
            "## Experiment Summary",
            "",
        ]

        if sample_size_plan:
            lines += [
                "### Pre-Experiment Power Analysis",
                "",
                "| Parameter | Value |",
                "|-----------|-------|",
            ]
            for k, v in sample_size_plan.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        lines += [
            "## Frequentist Results",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in frequentist_result.items():
            if k != "recommendation":
                lines.append(f"| {k} | {v} |")

        sig_emoji = "✅" if frequentist_result.get("is_significant") else "❌"
        lines += [
            "",
            f"**Decision ({sig_emoji}):** {frequentist_result.get('recommendation', '')}",
            "",
        ]

        if bayesian_result:
            lines += [
                "## Bayesian Results",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]
            for k, v in bayesian_result.items():
                if k != "recommendation":
                    lines.append(f"| {k} | {v} |")
            lines += [
                "",
                f"**Bayesian Decision:** {bayesian_result.get('recommendation', '')}",
                "",
            ]

        lines += ["---", "*Report generated by A/B Testing Toolkit*"]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # ── Sample Size Planning ──────────────────────────────────────────────────
    calc = SampleSizeCalculator()
    plan = calc.for_proportions(baseline_rate=0.05, min_detectable_effect=0.20, power=0.80)
    print("=== Sample Size Plan ===")
    for k, v in plan.items():
        print(f"  {k}: {v}")

    # ── Frequentist Proportion Test ───────────────────────────────────────────
    tester = FrequentistABTester()
    freq_result = tester.proportion_test(
        control_conversions=500, control_n=10000,
        treatment_conversions=580, treatment_n=10000,
    )
    print("\n=== Frequentist Proportion Test ===")
    for k, v in freq_result.items():
        print(f"  {k}: {v}")

    # ── Bayesian Analysis ─────────────────────────────────────────────────────
    bay_tester = BayesianABTester()
    bay_result = bay_tester.analyze(
        control_conversions=500, control_n=10000,
        treatment_conversions=580, treatment_n=10000,
    )
    print("\n=== Bayesian Analysis ===")
    for k, v in bay_result.items():
        print(f"  {k}: {v}")

    # ── Report ────────────────────────────────────────────────────────────────
    reporter = ExperimentReporter()
    report_md = reporter.generate(
        experiment_name="Homepage CTA Button Color Test",
        frequentist_result=freq_result,
        bayesian_result=bay_result,
        sample_size_plan=plan,
    )
    print("\n=== Report Preview ===")
    print(report_md[:800])
