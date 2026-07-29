# app.py
"""Streamlit app: A/B Testing & Experimentation Analysis.

Three modes, as tabs:
  - Real data: the Cookie Cats mobile-game retention experiment (Kaggle).
  - Simulated examples: labeled-synthetic scenarios illustrating distinct
    verdict types (borderline, clear win, inconclusive, harmful).
  - Multi-variant example (simulated): a multi-candidate search-algorithm
    test with multiple-comparison correction.

The significance level (alpha) is a single sidebar control shared by every
tab, so changing it actually re-runs the real z-test/power/verdict math
rather than just restyling a fixed result.
"""

from __future__ import annotations

import streamlit as st

from analysis.cookie_cats_analysis import analyze_retention, load_data
from analysis.verdict import recommend
from data.example_tests import ALL_EXAMPLES, EXAMPLES_BY_KEY, SEARCH_ALGORITHM
from stats.frequentist import two_proportion_z_test, welch_t_test
from stats.multivariate import VariantData, run_multivariate_test
from stats.power import current_power
from visualization.charts import (
    conversion_rate_comparison_chart,
    lift_confidence_interval_chart,
    multivariant_bar_chart,
    p_value_comparison_chart,
)

VERDICT_STYLE = {
    "ship": (st.success, "✅", "SHIP"),
    "hold": (st.warning, "⏸️", "HOLD"),
    "no-ship": (st.error, "🚫", "NO-SHIP"),
}


@st.cache_data(show_spinner="Loading Cookie Cats dataset...")
def _load_cookie_cats_data():
    """Cached wrapper around analysis.cookie_cats_analysis.load_data.

    Avoids re-reading the 90k-row CSV and recomputing on every widget
    interaction. Wraps an imported function, so it can't be decorated
    directly with @st.cache_data at its own definition site.
    """
    return load_data()


def render_verdict(verdict) -> None:
    box_fn, icon, label = VERDICT_STYLE[verdict.decision]
    box_fn(f"**{icon} Verdict: {label}**")
    with st.expander("Show the statistical reasoning"):
        for reason in verdict.reasons:
            st.markdown(f"- {reason}")


st.set_page_config(page_title="A/B Testing & Experimentation", layout="wide")
st.title("A/B Testing & Experimentation Analysis")
st.caption(
    "Real hypothesis testing, power analysis, and ship/no-ship recommendations "
    "— every number below comes from code in this repo, not hand-typed figures."
)

alpha = st.sidebar.slider(
    "Significance level (α)",
    min_value=0.01,
    max_value=0.10,
    value=0.05,
    step=0.01,
    help="Shared across every tab — moving this actually re-runs the z-test, "
    "power calculation, and verdict logic at the new threshold.",
)
st.sidebar.caption(f"Testing at the {(1 - alpha) * 100:.0f}% confidence level.")

tab_real, tab_sim, tab_multi = st.tabs(
    ["📊 Real Data: Cookie Cats", "🧪 Simulated Examples", "🔀 Multi-Variant (Simulated)"]
)

with tab_real:
    st.subheader("Cookie Cats: Gate 30 vs. Gate 40")
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
            "Try the **\"Simulated Examples\"** or **\"Multi-Variant\"** tabs "
            f"instead — they don't require Kaggle access.\n\n*Details: {e}*"
        )
        st.stop()
    st.write(f"Loaded **{len(df):,}** real players.")

    metric = st.radio("Retention metric", ["retention_1", "retention_7"], horizontal=True)
    z, power, verdict = analyze_retention(df, metric, alpha=alpha)

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Control (gate_30) rate", f"{z.control_rate*100:.2f}%")
        col2.metric(
            "Variant (gate_40) rate",
            f"{z.variant_rate*100:.2f}%",
            delta=f"{z.absolute_lift*100:+.2f} pp",
        )
        col3.metric("p-value", f"{z.p_value:.4f}", delta=f"achieved power {power.power:.2f}")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            conversion_rate_comparison_chart(
                "Gate 30 (Control)", "Gate 40 (Variant)",
                z.control_rate, z.variant_rate, z.control_n, z.variant_n, alpha=alpha,
            ),
            use_container_width=True,
        )
    with chart_col2:
        st.plotly_chart(
            lift_confidence_interval_chart(z.absolute_lift, z.ci_low, z.ci_high, z.is_significant),
            use_container_width=True,
        )

    render_verdict(verdict)

with tab_sim:
    st.subheader("Simulated Example Scenarios")
    st.info(
        f"**Simulated data** — these {len(ALL_EXAMPLES)} scenarios use "
        "seeded-RNG synthetic counts, chosen to illustrate distinct "
        "real-world verdicts. Not real experiment data.",
        icon="🧪",
    )
    example_key = st.selectbox(
        "Scenario", options=[ex.key for ex in ALL_EXAMPLES],
        format_func=lambda k: f"{EXAMPLES_BY_KEY[k].name} — {EXAMPLES_BY_KEY[k].tagline}",
    )
    ex = EXAMPLES_BY_KEY[example_key]
    st.markdown(f"**{ex.name}**")
    st.caption(ex.narrative)

    z = two_proportion_z_test(ex.control_conversions, ex.control_n, ex.variant_conversions, ex.variant_n, alpha=alpha)
    power = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n, alpha=alpha)
    verdict = recommend(z, power)

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric(ex.control_label, f"{z.control_rate*100:.2f}%")
        col2.metric(
            ex.variant_label, f"{z.variant_rate*100:.2f}%", delta=f"{z.absolute_lift*100:+.2f} pp"
        )
        col3.metric("p-value", f"{z.p_value:.4f}", delta=f"achieved power {power.power:.2f}")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            conversion_rate_comparison_chart(
                ex.control_label, ex.variant_label, z.control_rate, z.variant_rate, z.control_n, z.variant_n, alpha=alpha
            ),
            use_container_width=True,
        )
    with chart_col2:
        st.plotly_chart(
            lift_confidence_interval_chart(z.absolute_lift, z.ci_low, z.ci_high, z.is_significant),
            use_container_width=True,
        )

    if ex.control_values is not None:
        t = welch_t_test(ex.control_values, ex.variant_values, alpha=alpha)
        st.write(
            f"**{ex.continuous_metric_name}**: control mean ${t.control_mean:.2f}, "
            f"variant mean ${t.variant_mean:.2f}, p={t.p_value:.4f}"
        )

    render_verdict(verdict)

with tab_multi:
    st.subheader(SEARCH_ALGORITHM.name)
    st.info("**Simulated data** — 4 candidate algorithms tested against control at once.", icon="🧪")
    st.caption(SEARCH_ALGORITHM.narrative)

    control = VariantData(
        name=SEARCH_ALGORITHM.control_label, n=SEARCH_ALGORITHM.control_n, conversions=SEARCH_ALGORITHM.control_conversions
    )
    variants = [
        VariantData(name=name, n=n, conversions=conv)
        for name, n, conv in zip(
            SEARCH_ALGORITHM.variant_labels, SEARCH_ALGORITHM.variant_ns, SEARCH_ALGORITHM.variant_conversions
        )
    ]
    result = run_multivariate_test(control, variants, alpha=alpha)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            multivariant_bar_chart(
                control.name, control.rate,
                [c.variant_name for c in result.comparisons],
                [c.variant_rate for c in result.comparisons],
                [c.significant_bh for c in result.comparisons],
            ),
            use_container_width=True,
        )
    with chart_col2:
        st.plotly_chart(
            p_value_comparison_chart(
                [c.variant_name for c in result.comparisons],
                [c.p_value_raw for c in result.comparisons],
                [c.p_value_bonferroni for c in result.comparisons],
                [c.p_value_bh for c in result.comparisons],
                alpha=alpha,
            ),
            use_container_width=True,
        )

    if result.winner:
        st.success(f"**✅ Recommended winner: {result.winner}**")
    else:
        st.warning("**⏸️ No variant reached significance after correction — hold.**")
