"""Sequential testing with an O'Brien-Fleming alpha-spending boundary.

Why sequential testing matters: if you peek at a fixed-horizon A/B test
every day and stop as soon as p < 0.05, you inflate the true false-positive
rate far above 5% ("peeking problem" / optional stopping). Group-sequential
/ alpha-spending methods fix this by using a boundary that is stricter early
in the test (when information is low) and relaxes toward the nominal
critical value as the test accumulates its planned sample size. This lets
you monitor results every day and stop early *validly* if the effect is
large and obvious, without inflating alpha.

We use the classic O'Brien-Fleming spending function (Lan & DeMets 1983;
O'Brien & Fleming 1979), expressed as a function of the "information
fraction" t = current_n / planned_max_n (0 < t <= 1):

    alpha_spent(t) = 2 * (1 - Phi( z_(1-alpha/2) / sqrt(t) ))

Inverting this gives the z-statistic boundary that must be exceeded to
declare significance at information fraction t while preserving the
overall two-sided alpha:

    z_boundary(t) = z_(1-alpha/2) / sqrt(t)

As t -> 1 (test fully enrolled), the boundary converges to the ordinary
fixed-sample critical value z_(1-alpha/2) (e.g. 1.96 for alpha=0.05). For
t < 1, the boundary is higher, i.e. stricter -- this is what makes early
peeking safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats


def obf_boundary(t: np.ndarray | float, alpha: float = 0.05) -> np.ndarray | float:
    """O'Brien-Fleming z-statistic boundary at information fraction t.

        z_boundary(t) = z_(1-alpha/2) / sqrt(t)
    """
    t = np.asarray(t, dtype=float)
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        boundary = z_alpha / np.sqrt(t)
    return boundary


def obf_alpha_spent(t: np.ndarray | float, alpha: float = 0.05) -> np.ndarray | float:
    """Cumulative alpha "spent" by information fraction t.

        alpha_spent(t) = 2 * (1 - Phi(z_(1-alpha/2) / sqrt(t)))
    """
    boundary = obf_boundary(t, alpha)
    return 2 * (1 - sp_stats.norm.cdf(boundary))


@dataclass
class SequentialDayResult:
    day: int
    control_n: int
    control_conversions: int
    variant_n: int
    variant_conversions: int
    information_fraction: float
    z_stat: float
    boundary: float
    crossed: bool


@dataclass
class SequentialResult:
    days: list[SequentialDayResult] = field(default_factory=list)
    planned_n_per_group: int = 0
    alpha: float = 0.05
    first_crossing_day: int | None = None

    @property
    def has_crossed(self) -> bool:
        return self.first_crossing_day is not None


def run_sequential_monitoring(
    daily_control_n: list[int],
    daily_control_conversions: list[int],
    daily_variant_n: list[int],
    daily_variant_conversions: list[int],
    planned_n_per_group: int,
    alpha: float = 0.05,
) -> SequentialResult:
    """Run day-by-day sequential monitoring against an O'Brien-Fleming boundary.

    Inputs are *daily incremental* counts (new users / new conversions added
    each day). Cumulative totals are computed internally. `planned_n_per_group`
    is the sample size per arm at which the test is considered "fully
    enrolled" (information fraction t=1); it should come from the standard
    power-analysis sample-size calculation for the MDE you designed the test
    to detect.

    At each day, the two-proportion z-statistic is computed on the
    cumulative data (pooled-variance z-test, matching `frequentist.two_proportion_z_test`)
    and compared against the O'Brien-Fleming boundary at that day's
    information fraction. The test is flagged as "crossed" (safe to stop
    and ship) the first day |z| exceeds the boundary.
    """
    n = len(daily_control_n)
    if not (len(daily_control_conversions) == len(daily_variant_n) == len(daily_variant_conversions) == n):
        raise ValueError("All daily input lists must be the same length.")
    if planned_n_per_group <= 0:
        raise ValueError("planned_n_per_group must be positive.")

    result = SequentialResult(planned_n_per_group=planned_n_per_group, alpha=alpha)

    cum_control_n = 0
    cum_control_conv = 0
    cum_variant_n = 0
    cum_variant_conv = 0

    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)

    for day_idx in range(n):
        cum_control_n += daily_control_n[day_idx]
        cum_control_conv += daily_control_conversions[day_idx]
        cum_variant_n += daily_variant_n[day_idx]
        cum_variant_conv += daily_variant_conversions[day_idx]

        # Information fraction: how far along we are toward the planned
        # per-group sample size, averaged across the two arms, capped at 1.
        avg_n = (cum_control_n + cum_variant_n) / 2
        t = min(avg_n / planned_n_per_group, 1.0)
        t = max(t, 1e-6)  # avoid divide-by-zero on day 1 with tiny n

        p1 = cum_control_conv / cum_control_n if cum_control_n > 0 else 0.0
        p2 = cum_variant_conv / cum_variant_n if cum_variant_n > 0 else 0.0
        p_pool = (
            (cum_control_conv + cum_variant_conv) / (cum_control_n + cum_variant_n)
            if (cum_control_n + cum_variant_n) > 0
            else 0.0
        )
        se_pool = np.sqrt(
            p_pool * (1 - p_pool) * (1 / cum_control_n + 1 / cum_variant_n)
        ) if cum_control_n > 0 and cum_variant_n > 0 else np.nan

        z_stat = (p2 - p1) / se_pool if se_pool and se_pool > 0 else 0.0
        boundary = float(z_alpha / np.sqrt(t))
        crossed = abs(z_stat) >= boundary

        if crossed and result.first_crossing_day is None:
            result.first_crossing_day = day_idx + 1

        result.days.append(
            SequentialDayResult(
                day=day_idx + 1,
                control_n=cum_control_n,
                control_conversions=cum_control_conv,
                variant_n=cum_variant_n,
                variant_conversions=cum_variant_conv,
                information_fraction=float(t),
                z_stat=float(z_stat),
                boundary=boundary,
                crossed=bool(crossed),
            )
        )

    return result
