# app.py
"""Streamlit app: A/B Testing & Experimentation Analysis.

Three modes:
  - Real data: the Cookie Cats mobile-game retention experiment (Kaggle).
  - Simulated examples: labeled-synthetic scenarios illustrating distinct
    verdict types (borderline, clear win, inconclusive, harmful).
  - Multi-variant example (simulated): a multi-candidate search-algorithm
    test with multiple-comparison correction.
"""

from __future__ import annotations

import streamlit as st

from analysis.cookie_cats_analysis import analyze_retention, load_data
from analysis.verdict import recommend
from data.example_tests import ALL_EXAMPLES, EXAMPLES_BY_KEY, SEARCH_ALGORITHM
from stats.frequentist import two_proportion_z_test, welch_t_test
from stats.multivariate import run_multivariate_test
from stats.power import current_power
from visualization.charts import (
    conversion_rate_comparison_chart,
    lift_confidence_interval_chart,
    multivariant_bar_chart,
    p_value_comparison_chart,
)

@st.cache_data(show_spinner="Loading Cookie Cats dataset...")
def _load_cookie_cats_data():
    """Cached wrapper around analysis.cookie_cats_analysis.load_data.

    Avoids re-reading the 90k-row CSV and recomputing on every widget
    interaction. Wraps an imported function, so it can't be decorated
    directly with @st.cache_data at its own definition site.
    """
    return load_data()


st.set_page_config(page_title="A/B Testing & Experimentation", layout="wide")
st.title("A/B Testing & Experimentation Analysis")
st.caption(
    "Real hypothesis testing, power analysis, and ship/no-ship recommendations "
    "— every number below comes from code in this repo, not hand-typed figures."
)

mode = st.sidebar.radio(
    "Data source",
    ["Real data: Cookie Cats (Kaggle)", "Simulated examples", "Multi-variant example (simulated)"],
)

if mode == "Real data: Cookie Cats (Kaggle)":
    st.header("Cookie Cats: Gate 30 vs. Gate 40")
    st.markdown(
        "**Real dataset** — [Cookie Cats mobile game A/B test]"
        "(https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats), "
        "not simulated. Does moving the level-30 paywall gate to level 40 "
        "change player retention?"
    )
    try:
        df = _load_cookie_cats_data()
    except Exception as e:
        st.error(
            "**Could not load the real Cookie Cats dataset.** This mode "
            "downloads data via the Kaggle API, which requires Kaggle "
            "credentials (`~/.kaggle/kaggle.json`) that aren't available in "
            "this deployment.\n\n"
            "Try the **\"Simulated examples\"** or **\"Multi-variant example "
            "(simulated)\"** modes instead — they don't require Kaggle "
            f"access.\n\n*Details: {e}*"
        )
        st.stop()
    st.write(f"Loaded **{len(df):,}** real players.")

    metric = st.selectbox("Retention metric", ["retention_1", "retention_7"])
    z, power, verdict = analyze_retention(df, metric)

    col1, col2, col3 = st.columns(3)
    col1.metric("Control (gate_30) rate", f"{z.control_rate*100:.2f}%")
    col2.metric("Variant (gate_40) rate", f"{z.variant_rate*100:.2f}%")
    col3.metric("p-value", f"{z.p_value:.4f}")

    st.plotly_chart(
        conversion_rate_comparison_chart(
            "Gate 30 (Control)", "Gate 40 (Variant)",
            z.control_rate, z.variant_rate, z.control_n, z.variant_n,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        lift_confidence_interval_chart(z.absolute_lift, z.ci_low, z.ci_high, z.is_significant),
        use_container_width=True,
    )
    st.write(f"Achieved power (post-hoc, from observed effect): **{power.power:.2f}**")

    verdict_color = {"ship": "green", "hold": "orange", "no-ship": "red"}[verdict.decision]
    st.markdown(f"### Verdict: :{verdict_color}[{verdict.decision.upper()}]")
    for reason in verdict.reasons:
        st.markdown(f"- {reason}")

elif mode == "Simulated examples":
    st.header("Simulated Example Scenarios")
    st.info(
        f"**Simulated data** — these {len(ALL_EXAMPLES)} scenarios use "
        "seeded-RNG synthetic counts, chosen to illustrate distinct "
        "real-world verdicts. Not real experiment data."
    )
    example_key = st.selectbox(
        "Scenario", options=[ex.key for ex in ALL_EXAMPLES],
        format_func=lambda k: EXAMPLES_BY_KEY[k].name,
    )
    ex = EXAMPLES_BY_KEY[example_key]
    st.subheader(ex.name)
    st.caption(ex.tagline)
    st.write(ex.narrative)

    z = two_proportion_z_test(ex.control_conversions, ex.control_n, ex.variant_conversions, ex.variant_n)
    power = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n)
    verdict = recommend(z, power)

    col1, col2, col3 = st.columns(3)
    col1.metric(ex.control_label, f"{z.control_rate*100:.2f}%")
    col2.metric(ex.variant_label, f"{z.variant_rate*100:.2f}%")
    col3.metric("p-value", f"{z.p_value:.4f}")

    st.plotly_chart(
        conversion_rate_comparison_chart(ex.control_label, ex.variant_label, z.control_rate, z.variant_rate, z.control_n, z.variant_n),
        use_container_width=True,
    )
    st.plotly_chart(
        lift_confidence_interval_chart(z.absolute_lift, z.ci_low, z.ci_high, z.is_significant),
        use_container_width=True,
    )

    if ex.control_values is not None:
        t = welch_t_test(ex.control_values, ex.variant_values)
        st.write(f"**{ex.continuous_metric_name}**: control mean ${t.control_mean:.2f}, variant mean ${t.variant_mean:.2f}, p={t.p_value:.4f}")

    verdict_color = {"ship": "green", "hold": "orange", "no-ship": "red"}[verdict.decision]
    st.markdown(f"### Verdict: :{verdict_color}[{verdict.decision.upper()}]")
    for reason in verdict.reasons:
        st.markdown(f"- {reason}")

else:
    st.header(SEARCH_ALGORITHM.name)
    st.info("**Simulated data** — 4 candidate algorithms tested against control at once.")
    st.write(SEARCH_ALGORITHM.narrative)

    from stats.multivariate import VariantData

    control = VariantData(name=SEARCH_ALGORITHM.control_label, n=SEARCH_ALGORITHM.control_n, conversions=SEARCH_ALGORITHM.control_conversions)
    variants = [
        VariantData(name=name, n=n, conversions=conv)
        for name, n, conv in zip(SEARCH_ALGORITHM.variant_labels, SEARCH_ALGORITHM.variant_ns, SEARCH_ALGORITHM.variant_conversions)
    ]
    result = run_multivariate_test(control, variants)

    st.plotly_chart(
        multivariant_bar_chart(
            control.name, control.rate,
            [c.variant_name for c in result.comparisons],
            [c.variant_rate for c in result.comparisons],
            [c.significant_bh for c in result.comparisons],
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        p_value_comparison_chart(
            [c.variant_name for c in result.comparisons],
            [c.p_value_raw for c in result.comparisons],
            [c.p_value_bonferroni for c in result.comparisons],
            [c.p_value_bh for c in result.comparisons],
        ),
        use_container_width=True,
    )

    if result.winner:
        st.markdown(f"### Recommended winner: :green[{result.winner}]")
    else:
        st.markdown("### No variant reached significance after correction — :orange[hold]")
