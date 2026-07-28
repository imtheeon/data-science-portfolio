"""Multi-variant ("A/B/n") testing with multiple-comparison corrections.

When testing more than one variant against control at once, running many
independent hypothesis tests inflates the family-wise false-positive rate:
with m independent tests each at alpha=0.05, the chance of at least one
false positive is 1 - (1-0.05)^m, e.g. ~23% for m=5 tests, not 5%. This
module runs all pairwise-vs-control z-tests and then applies two standard
corrections:

  * Bonferroni  -- controls the family-wise error rate (FWER) by requiring
                   p < alpha/m (equivalently, adjusted p = min(p*m, 1)).
                   Conservative but simple and safe.
  * Benjamini-Hochberg (BH) -- controls the false discovery rate (FDR)
                   instead of FWER, and is less conservative (more power)
                   when running several comparisons at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stats.frequentist import two_proportion_z_test


@dataclass
class VariantData:
    name: str
    n: int
    conversions: int

    @property
    def rate(self) -> float:
        return self.conversions / self.n if self.n else 0.0


@dataclass
class PairwiseResult:
    variant_name: str
    control_rate: float
    variant_rate: float
    absolute_lift: float
    relative_lift: float
    z_stat: float
    p_value_raw: float
    p_value_bonferroni: float
    p_value_bh: float
    significant_bonferroni: bool
    significant_bh: bool


@dataclass
class MultivariateResult:
    control: VariantData
    variants: list[VariantData]
    comparisons: list[PairwiseResult]
    alpha: float
    winner: str | None  # name of recommended winning variant, or None


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> np.ndarray:
    """Bonferroni-adjusted p-values: p_adj_i = min(p_i * m, 1), m = number of tests."""
    m = len(p_values)
    p = np.asarray(p_values, dtype=float)
    return np.minimum(p * m, 1.0)


def benjamini_hochberg_correction(p_values: list[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), controlling FDR.

    Procedure
    ---------
    1. Sort the m p-values ascending: p_(1) <= p_(2) <= ... <= p_(m).
    2. Raw adjusted value for rank i (1-indexed): p_(i) * m / i.
    3. Enforce monotonicity by taking a running minimum from the largest
       rank down to the smallest (a BH q-value cannot exceed the q-value
       of a less-significant result).
    4. Map back to the original order.

    A hypothesis i is rejected at level alpha if its BH q-value <= alpha.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]

    raw_adjusted = ranked * m / (np.arange(m) + 1)
    # enforce monotone non-increasing when scanning from largest rank to smallest
    adjusted_sorted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    q_values = np.empty(m)
    q_values[order] = adjusted_sorted
    return q_values


def run_multivariate_test(
    control: VariantData,
    variants: list[VariantData],
    alpha: float = 0.05,
) -> MultivariateResult:
    """Run pairwise-vs-control z-tests for up to 5 variants with corrections."""
    if not (1 <= len(variants) <= 5):
        raise ValueError("Support 1 to 5 variants (excluding control).")

    raw_results = [
        two_proportion_z_test(control.conversions, control.n, v.conversions, v.n, alpha=alpha)
        for v in variants
    ]
    p_values = [r.p_value for r in raw_results]

    p_bonf = bonferroni_correction(p_values, alpha)
    p_bh = benjamini_hochberg_correction(p_values, alpha)

    comparisons = []
    for v, r, pb, pq in zip(variants, raw_results, p_bonf, p_bh):
        comparisons.append(
            PairwiseResult(
                variant_name=v.name,
                control_rate=r.control_rate,
                variant_rate=r.variant_rate,
                absolute_lift=r.absolute_lift,
                relative_lift=r.relative_lift,
                z_stat=r.z_stat,
                p_value_raw=r.p_value,
                p_value_bonferroni=float(pb),
                p_value_bh=float(pq),
                significant_bonferroni=bool(pb < alpha),
                significant_bh=bool(pq < alpha),
            )
        )

    # Winner recommendation: among variants significant under the more
    # powerful BH correction with a positive lift, pick the highest rate.
    # Fall back to "no significant winner" if none qualify.
    significant_positive = [c for c in comparisons if c.significant_bh and c.absolute_lift > 0]
    winner = None
    if significant_positive:
        winner = max(significant_positive, key=lambda c: c.variant_rate).variant_name

    return MultivariateResult(
        control=control,
        variants=variants,
        comparisons=comparisons,
        alpha=alpha,
        winner=winner,
    )
