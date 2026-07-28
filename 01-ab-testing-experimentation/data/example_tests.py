"""Five pre-loaded example A/B test scenarios for instant demoing.

Each example is a self-contained, internally consistent dataset: conversion
counts, an optional continuous metric (for the Welch t-test path), and
synthetic-but-reproducible day-by-day traffic (for the Sequential Monitor
tab). Numbers were chosen deliberately to reproduce five distinct real-world
verdicts a product analyst runs into constantly:

    1. Checkout Button Color   -> barely significant / borderline (p ~ 0.049)
    2. Pricing Page Redesign   -> clear, obvious winner (p < 0.001)
    3. Email Subject Line      -> not significant, need more data (p ~ 0.5)
    4. Onboarding Flow         -> the variant is actually HARMFUL (control wins)
    5. Search Algorithm        -> multi-variant with mixed significant/null/harmful results

Daily data is generated with a seeded RNG so re-running always produces the
same numbers (reproducible demo), while still looking like real noisy daily
traffic rather than a flat hardcoded line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DailyData:
    control_n: list[int]
    control_conversions: list[int]
    variant_n: list[int]
    variant_conversions: list[int]


@dataclass
class ExampleTest:
    key: str
    name: str
    tagline: str
    narrative: str
    control_label: str
    variant_label: str
    control_n: int
    control_conversions: int
    variant_n: int
    variant_conversions: int
    daily_data: DailyData
    control_values: list[float] | None = None  # optional continuous metric
    variant_values: list[float] | None = None
    continuous_metric_name: str | None = None


@dataclass
class MultiVariantExample:
    key: str
    name: str
    tagline: str
    narrative: str
    control_label: str
    control_n: int
    control_conversions: int
    variant_labels: list[str]
    variant_ns: list[int]
    variant_conversions: list[int]


def _split_into_daily(total_n: int, total_conversions: int, days: int, seed: int) -> tuple[list[int], list[int]]:
    """Split a final (n, conversions) total into `days` daily increments that
    sum exactly to the totals, using a seeded multinomial draw so the demo
    is reproducible but still looks like organic daily traffic/conversions."""
    rng = np.random.default_rng(seed)
    # Traffic: roughly even daily probability with a little noise
    weights = rng.dirichlet(np.full(days, 20.0))
    daily_n = rng.multinomial(total_n, weights)
    daily_n = np.maximum(daily_n, 1)  # avoid zero-traffic days
    # rescale to still sum to total_n after the floor above
    while daily_n.sum() != total_n:
        diff = total_n - daily_n.sum()
        idx = rng.integers(0, days)
        if diff > 0:
            daily_n[idx] += 1
        elif daily_n[idx] > 1:
            daily_n[idx] -= 1

    # Conversions: distribute proportionally to daily traffic via multinomial
    conv_weights = daily_n / daily_n.sum()
    daily_conv = rng.multinomial(total_conversions, conv_weights)
    daily_conv = np.minimum(daily_conv, daily_n)

    return daily_n.tolist(), daily_conv.tolist()


def _make_daily_data(
    control_n: int, control_conv: int, variant_n: int, variant_conv: int, days: int, seed: int
) -> DailyData:
    c_n, c_conv = _split_into_daily(control_n, control_conv, days, seed)
    v_n, v_conv = _split_into_daily(variant_n, variant_conv, days, seed + 1)
    return DailyData(control_n=c_n, control_conversions=c_conv, variant_n=v_n, variant_conversions=v_conv)


def _revenue_samples(mean: float, std: float, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=mean, scale=std, size=n)
    return np.clip(samples, 0, None).round(2).tolist()


CHECKOUT_BUTTON_COLOR = ExampleTest(
    key="checkout_button_color",
    name="Checkout Button Color",
    tagline="Borderline / barely significant",
    narrative=(
        "Changed the 'Complete Purchase' button from blue to green on the checkout page. "
        "The variant shows a positive lift, but it's a coin-flip call: the p-value lands "
        "right at the edge of the conventional 0.05 threshold. This is the case that trips "
        "up teams who eyeball 'green means ship' without checking how fragile that result is."
    ),
    control_label="Blue Button (Control)",
    variant_label="Green Button (Variant)",
    control_n=1000,
    control_conversions=100,
    variant_n=1000,
    variant_conversions=128,
    daily_data=_make_daily_data(1000, 100, 1000, 128, days=21, seed=101),
)

PRICING_PAGE_REDESIGN = ExampleTest(
    key="pricing_page_redesign",
    name="Pricing Page Redesign",
    tagline="Clear, obvious winner",
    narrative=(
        "Redesigned the pricing page with clearer tier comparisons and social proof. "
        "Conversion to paid plan jumped from 8.0% to 10.4%, and with 5,000 users per arm "
        "the effect is overwhelming -- this is what a genuinely clear win looks like "
        "statistically, not just directionally."
    ),
    control_label="Original Pricing Page (Control)",
    variant_label="Redesigned Pricing Page (Variant)",
    control_n=5000,
    control_conversions=400,
    variant_n=5000,
    variant_conversions=520,
    daily_data=_make_daily_data(5000, 400, 5000, 520, days=30, seed=201),
    control_values=_revenue_samples(mean=42.0, std=18.0, n=400, seed=203),
    variant_values=_revenue_samples(mean=51.0, std=20.0, n=520, seed=204),
    continuous_metric_name="Revenue per Converting User ($)",
)

EMAIL_SUBJECT_LINE = ExampleTest(
    key="email_subject_line",
    name="Email Subject Line",
    tagline="Not significant -- need more data",
    narrative=(
        "Tested a curiosity-driven subject line against the standard, descriptive one for "
        "the weekly product newsletter. Open rate looks a little higher for the variant, "
        "but the sample is small and the gap is well within the range you'd expect from "
        "noise alone. The honest verdict here is 'inconclusive,' not 'ship it.'"
    ),
    control_label="Standard Subject Line (Control)",
    variant_label="Curiosity Subject Line (Variant)",
    control_n=300,
    control_conversions=45,
    variant_n=300,
    variant_conversions=51,
    daily_data=_make_daily_data(300, 45, 300, 51, days=7, seed=301),
)

ONBOARDING_FLOW = ExampleTest(
    key="onboarding_flow",
    name="Onboarding Flow",
    tagline="Harmful variant -- control wins",
    narrative=(
        "Replaced the 5-step onboarding wizard with a single condensed screen to reduce "
        "friction. It backfired: completion rate dropped from 15.0% to 12.75%, a "
        "statistically significant regression. Without a rigorous test, this variant could "
        "easily have shipped on the (wrong) assumption that 'fewer steps is always better.'"
    ),
    control_label="5-Step Wizard (Control)",
    variant_label="Condensed Single Screen (Variant)",
    control_n=2000,
    control_conversions=300,
    variant_n=2000,
    variant_conversions=255,
    daily_data=_make_daily_data(2000, 300, 2000, 255, days=14, seed=401),
)

SEARCH_ALGORITHM = MultiVariantExample(
    key="search_algorithm",
    name="Search Algorithm",
    tagline="Multi-variant, mixed results",
    narrative=(
        "Four candidate ranking algorithms were tested at once against the current "
        "production search ranker, measuring click-through on the first result. Running "
        "four comparisons at once inflates the false-positive rate if left uncorrected -- "
        "this example is built specifically to show Bonferroni and Benjamini-Hochberg "
        "disagreeing on one variant (LLM Query Rewrite), which is exactly the kind of "
        "judgment call a multi-variant test forces you to make explicit."
    ),
    control_label="Current Algorithm (Control)",
    control_n=2000,
    control_conversions=300,
    variant_labels=["BM25 Tuning", "Neural Reranker", "Hybrid Search", "LLM Query Rewrite"],
    variant_ns=[2000, 2000, 2000, 2000],
    variant_conversions=[330, 370, 280, 250],
)


ALL_EXAMPLES: list[ExampleTest] = [
    CHECKOUT_BUTTON_COLOR,
    PRICING_PAGE_REDESIGN,
    EMAIL_SUBJECT_LINE,
    ONBOARDING_FLOW,
]

EXAMPLES_BY_KEY: dict[str, ExampleTest] = {ex.key: ex for ex in ALL_EXAMPLES}
