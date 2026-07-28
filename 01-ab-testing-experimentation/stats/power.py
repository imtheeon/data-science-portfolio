"""Sample size and statistical power calculations for two-proportion tests.

All formulas follow the standard normal-approximation approach used by
widely-cited practitioner tools (e.g. Evan Miller's sample size calculator)
and textbook treatments of power analysis for two-sample proportion tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats as sp_stats


@dataclass
class SampleSizeResult:
    baseline_rate: float
    target_rate: float
    mde_absolute: float
    alpha: float
    power: float
    n_per_group: int
    total_n: int


@dataclass
class PowerResult:
    baseline_rate: float
    observed_rate: float
    control_n: int
    variant_n: int
    alpha: float
    power: float


@dataclass
class DaysToSignificanceResult:
    n_per_group_required: int
    n_per_group_current: int
    daily_traffic_per_group: float
    days_remaining: float
    already_reached: bool


def required_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    relative: bool = True,
) -> SampleSizeResult:
    """Required per-group sample size to detect `mde` at the given alpha/power.

    Formula
    -------
    Let p1 = baseline_rate and p2 = the target rate under the alternative
    (p2 = p1 * (1 + mde) if `relative`, else p2 = p1 + mde).

        z_alpha/2 = Phi^-1(1 - alpha/2)     (two-sided critical value)
        z_beta    = Phi^-1(power)

        n_per_group = (z_alpha/2 + z_beta)^2 * (p1*(1-p1) + p2*(1-p2))
                      -----------------------------------------------
                                    (p2 - p1)^2

    This is the standard sample-size formula for a two-sample z-test on
    proportions with equal group sizes, using the (unpooled) variances of
    each arm under the alternative hypothesis.
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be strictly between 0 and 1.")

    target_rate = baseline_rate * (1 + mde) if relative else baseline_rate + mde
    if not (0 < target_rate < 1):
        raise ValueError("Resulting target_rate must be strictly between 0 and 1.")

    mde_absolute = target_rate - baseline_rate
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)

    numerator = (z_alpha + z_beta) ** 2 * (
        baseline_rate * (1 - baseline_rate) + target_rate * (1 - target_rate)
    )
    n_per_group = numerator / (mde_absolute ** 2)
    n_per_group_int = math.ceil(n_per_group)

    return SampleSizeResult(
        baseline_rate=baseline_rate,
        target_rate=target_rate,
        mde_absolute=mde_absolute,
        alpha=alpha,
        power=power,
        n_per_group=n_per_group_int,
        total_n=n_per_group_int * 2,
    )


def current_power(
    control_rate: float,
    variant_rate: float,
    control_n: int,
    variant_n: int,
    alpha: float = 0.05,
) -> PowerResult:
    """Statistical power actually achieved given the observed rates and n's.

    Formula
    -------
    delta = |variant_rate - control_rate|
    SE    = sqrt( p1*(1-p1)/n1 + p2*(1-p2)/n2 )   (unpooled SE, as under H1)
    z_alpha/2 = Phi^-1(1 - alpha/2)

    power = Phi( delta/SE - z_alpha/2 ) + Phi( -delta/SE - z_alpha/2 )

    (The second term is the negligible chance of detecting the effect in
    the "wrong" direction; it is included for exactness but is ~0 whenever
    delta is meaningfully large relative to SE.)
    """
    delta = abs(variant_rate - control_rate)
    se = math.sqrt(
        control_rate * (1 - control_rate) / control_n
        + variant_rate * (1 - variant_rate) / variant_n
    )
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)

    if se == 0:
        power = 1.0 if delta > 0 else 0.0
    else:
        power = sp_stats.norm.cdf(delta / se - z_alpha) + sp_stats.norm.cdf(
            -delta / se - z_alpha
        )

    return PowerResult(
        baseline_rate=control_rate,
        observed_rate=variant_rate,
        control_n=control_n,
        variant_n=variant_n,
        alpha=alpha,
        power=float(power),
    )


def days_to_significance(
    n_per_group_required: int,
    n_per_group_current: int,
    daily_traffic_per_group: float,
) -> DaysToSignificanceResult:
    """Estimate remaining days of data collection to reach the required n.

    Assumes daily traffic per group stays roughly constant. If the current
    sample size already meets or exceeds the requirement, `days_remaining`
    is 0 and `already_reached` is True.
    """
    if daily_traffic_per_group <= 0:
        raise ValueError("daily_traffic_per_group must be positive.")

    remaining_n = max(0, n_per_group_required - n_per_group_current)
    days = math.ceil(remaining_n / daily_traffic_per_group) if remaining_n > 0 else 0

    return DaysToSignificanceResult(
        n_per_group_required=n_per_group_required,
        n_per_group_current=n_per_group_current,
        daily_traffic_per_group=daily_traffic_per_group,
        days_remaining=days,
        already_reached=remaining_n == 0,
    )
