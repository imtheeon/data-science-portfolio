"""Frequentist hypothesis tests for A/B testing.

Implements the two-proportion z-test, Welch's unequal-variance t-test, and
the chi-square test of independence from first principles (i.e. the test
statistics and standard errors are computed directly from the formulas
below, not by calling a single all-in-one library function). scipy is used
only for the underlying probability distributions (norm, t, chi2) needed to
convert a statistic into a p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


@dataclass
class ZTestResult:
    """Result of a two-proportion z-test comparing control vs. variant."""

    control_rate: float
    variant_rate: float
    control_n: int
    variant_n: int
    control_conversions: int
    variant_conversions: int
    pooled_rate: float
    se_pooled: float          # standard error used for the z-statistic (H0: p1 == p2)
    se_unpooled: float        # standard error used for the CI on the difference
    z_stat: float
    p_value: float
    absolute_lift: float      # variant_rate - control_rate
    relative_lift: float      # absolute_lift / control_rate
    ci_low: float             # CI on the absolute lift (difference in proportions)
    ci_high: float
    alpha: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < self.alpha


def two_proportion_z_test(
    control_conversions: int,
    control_n: int,
    variant_conversions: int,
    variant_n: int,
    alpha: float = 0.05,
) -> ZTestResult:
    """Two-proportion (two-sided) z-test for conversion-rate data.

    Formulas
    --------
    p1 = control_conversions / control_n        (control conversion rate)
    p2 = variant_conversions / variant_n         (variant conversion rate)

    Pooled proportion (used under H0: p1 == p2, for the test statistic):
        p_pool = (control_conversions + variant_conversions) / (control_n + variant_n)
        SE_pool = sqrt(p_pool * (1 - p_pool) * (1/control_n + 1/variant_n))
        z = (p2 - p1) / SE_pool

    Two-sided p-value:
        p_value = 2 * (1 - Phi(|z|))   where Phi is the standard normal CDF

    Unpooled standard error (used for the confidence interval on the
    observed difference p2 - p1, since under the alternative the two
    proportions need not be equal):
        SE_unpooled = sqrt(p1*(1-p1)/control_n + p2*(1-p2)/variant_n)
        CI = (p2 - p1) +/- z_(1-alpha/2) * SE_unpooled
    """
    if control_n <= 0 or variant_n <= 0:
        raise ValueError("Sample sizes must be positive.")
    if not (0 <= control_conversions <= control_n):
        raise ValueError("control_conversions must be between 0 and control_n.")
    if not (0 <= variant_conversions <= variant_n):
        raise ValueError("variant_conversions must be between 0 and variant_n.")

    p1 = control_conversions / control_n
    p2 = variant_conversions / variant_n

    p_pool = (control_conversions + variant_conversions) / (control_n + variant_n)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / variant_n))

    if se_pool == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = (p2 - p1) / se_pool
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    se_unpooled = np.sqrt(
        p1 * (1 - p1) / control_n + p2 * (1 - p2) / variant_n
    )
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled

    relative_lift = diff / p1 if p1 > 0 else float("nan")

    return ZTestResult(
        control_rate=p1,
        variant_rate=p2,
        control_n=control_n,
        variant_n=variant_n,
        control_conversions=control_conversions,
        variant_conversions=variant_conversions,
        pooled_rate=p_pool,
        se_pooled=float(se_pool),
        se_unpooled=float(se_unpooled),
        z_stat=float(z),
        p_value=float(p_value),
        absolute_lift=float(diff),
        relative_lift=float(relative_lift),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        alpha=alpha,
    )


@dataclass
class TTestResult:
    """Result of Welch's unequal-variance t-test for continuous metrics."""

    control_mean: float
    variant_mean: float
    control_std: float
    variant_std: float
    control_n: int
    variant_n: int
    se: float
    t_stat: float
    df: float
    p_value: float
    absolute_lift: float
    relative_lift: float
    ci_low: float
    ci_high: float
    alpha: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < self.alpha


def welch_t_test(
    control_values: np.ndarray,
    variant_values: np.ndarray,
    alpha: float = 0.05,
) -> TTestResult:
    """Welch's t-test for two independent samples with unequal variance.

    Used for continuous metrics such as revenue-per-user or session duration
    where the two groups may not share a common variance (an assumption
    required by the classic pooled-variance t-test).

    Formulas
    --------
    mean1, mean2   : sample means
    s1^2, s2^2     : sample variances (ddof=1)
    n1, n2         : sample sizes

    SE = sqrt(s1^2/n1 + s2^2/n2)
    t  = (mean2 - mean1) / SE

    Welch-Satterthwaite degrees of freedom:
        df = (s1^2/n1 + s2^2/n2)^2 /
             ( (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) )

    Two-sided p-value from Student's t distribution with `df` degrees of
    freedom:
        p_value = 2 * (1 - T_df.cdf(|t|))

    CI on the difference in means:
        CI = (mean2 - mean1) +/- t_crit(df) * SE
    """
    control_values = np.asarray(control_values, dtype=float)
    variant_values = np.asarray(variant_values, dtype=float)

    n1, n2 = len(control_values), len(variant_values)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each sample needs at least 2 observations.")

    mean1, mean2 = control_values.mean(), variant_values.mean()
    var1, var2 = control_values.var(ddof=1), variant_values.var(ddof=1)

    se_sq = var1 / n1 + var2 / n2
    se = np.sqrt(se_sq)

    if se == 0:
        t_stat = 0.0
        df = n1 + n2 - 2
        p_value = 1.0
    else:
        t_stat = (mean2 - mean1) / se
        df = se_sq ** 2 / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )
        p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df))

    t_crit = sp_stats.t.ppf(1 - alpha / 2, df) if se > 0 else 0.0
    diff = mean2 - mean1
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se
    relative_lift = diff / mean1 if mean1 != 0 else float("nan")

    return TTestResult(
        control_mean=float(mean1),
        variant_mean=float(mean2),
        control_std=float(np.sqrt(var1)),
        variant_std=float(np.sqrt(var2)),
        control_n=n1,
        variant_n=n2,
        se=float(se),
        t_stat=float(t_stat),
        df=float(df),
        p_value=float(p_value),
        absolute_lift=float(diff),
        relative_lift=float(relative_lift),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        alpha=alpha,
    )


@dataclass
class ChiSquareResult:
    """Result of a chi-square test of independence on a 2xK contingency table."""

    chi2_stat: float
    dof: int
    p_value: float
    expected: np.ndarray
    observed: np.ndarray
    alpha: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < self.alpha


def chi_square_test(observed: np.ndarray, alpha: float = 0.05) -> ChiSquareResult:
    """Chi-square test of independence for a contingency table.

    Formulas
    --------
    For a table with row totals R_i, column totals C_j, and grand total N:
        E_ij = R_i * C_j / N                         (expected count)
        chi2 = sum over all cells of (O_ij - E_ij)^2 / E_ij
        dof  = (rows - 1) * (cols - 1)
        p_value = 1 - ChiSquare_dof.cdf(chi2)  ==  ChiSquare_dof.sf(chi2)

    Typical use: outcome is categorical with more than two levels (e.g.
    "bought tier A / tier B / nothing") and you want to test whether the
    distribution across categories differs by group (control vs variant).
    """
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 2:
        raise ValueError("observed must be a 2D array (rows=groups, cols=categories).")

    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()
    if grand_total <= 0:
        raise ValueError("Observed table must have a positive total count.")

    expected = row_totals @ col_totals / grand_total

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    chi2_stat = float(terms.sum())

    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_value = float(sp_stats.chi2.sf(chi2_stat, dof)) if dof > 0 else 1.0

    return ChiSquareResult(
        chi2_stat=chi2_stat,
        dof=dof,
        p_value=p_value,
        expected=expected,
        observed=observed,
        alpha=alpha,
    )
