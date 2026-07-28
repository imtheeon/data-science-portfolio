"""Interactive Plotly charts for the ABTestAnalyzer dashboard.

Color roles follow a fixed, validated categorical order (never reassigned by
rank) plus a small reserved status palette for verdicts, so a color always
means the same thing across every chart in the app:

    Categorical (identity, fixed order): blue, orange, aqua, yellow, magenta,
    green, violet, red -- slot 1 (blue) is always "control", slot 2 (orange)
    is always "variant" / "variant A", subsequent slots are additional
    variants in a multi-variant test, in the order they appear.

    Status (state, reserved -- never reused as a series color): good (green),
    warning (yellow), critical (red) -- used only for ship/hold/no-ship type
    verdicts, never for plain data identity.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats as sp_stats

# --- Fixed categorical palette (validated ordering, see dataviz skill) -----
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CONTROL_COLOR = CATEGORICAL[0]
VARIANT_COLOR = CATEGORICAL[1]

# --- Reserved status palette (never used for plain series identity) -------
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

# --- Chrome / ink ------------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
CHART_SURFACE = "#fcfcfb"

_BASE_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=CHART_SURFACE,
    plot_bgcolor=CHART_SURFACE,
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_PRIMARY, size=13),
    margin=dict(l=60, r=30, t=60, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _apply_axis_style(fig: go.Figure) -> None:
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BASELINE, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, linecolor=BASELINE, tickfont=dict(color=INK_MUTED))


def conversion_rate_comparison_chart(
    control_label: str,
    variant_label: str,
    control_rate: float,
    variant_rate: float,
    control_n: int,
    variant_n: int,
    alpha: float = 0.05,
) -> go.Figure:
    """Bar chart of control vs. variant conversion rate with a 95% Wald CI
    error bar on each bar (per-arm CI: p +/- z * sqrt(p(1-p)/n))."""
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    rates = [control_rate, variant_rate]
    ns = [control_n, variant_n]
    errors = [z_crit * np.sqrt(r * (1 - r) / n) if n > 0 else 0 for r, n in zip(rates, ns)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[control_label, variant_label],
            y=[r * 100 for r in rates],
            error_y=dict(type="data", array=[e * 100 for e in errors], color=INK_SECONDARY, thickness=1.5, width=6),
            marker_color=[CONTROL_COLOR, VARIANT_COLOR],
            text=[f"{r*100:.2f}%" for r in rates],
            textposition="outside",
            textfont=dict(color=INK_PRIMARY),
            width=0.5,
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Conversion Rate: Control vs. Variant",
        yaxis_title="Conversion rate (%)",
        showlegend=False,
        bargap=0.4,
    )
    _apply_axis_style(fig)
    return fig


def lift_confidence_interval_chart(
    absolute_lift: float, ci_low: float, ci_high: float, is_significant: bool
) -> go.Figure:
    """Horizontal point-estimate-with-CI chart of the lift, with a zero
    reference line. Color reflects the ship/no-ship verdict (status palette),
    not plain series identity, since this chart *is* the verdict."""
    color = STATUS_GOOD if (is_significant and absolute_lift > 0) else (
        STATUS_CRITICAL if (is_significant and absolute_lift < 0) else STATUS_WARNING
    )

    fig = go.Figure()
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color=BASELINE)
    fig.add_trace(
        go.Scatter(
            x=[absolute_lift * 100],
            y=["Lift"],
            mode="markers",
            marker=dict(color=color, size=16, line=dict(color=INK_PRIMARY, width=1)),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[(ci_high - absolute_lift) * 100],
                arrayminus=[(absolute_lift - ci_low) * 100],
                color=INK_SECONDARY,
                thickness=2,
                width=8,
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Lift (Absolute Percentage Points) with 95% Confidence Interval",
        xaxis_title="Absolute lift (percentage points)",
        height=220,
    )
    _apply_axis_style(fig)
    fig.update_yaxes(showgrid=False)
    return fig


def bayesian_posterior_chart(
    control_alpha: float,
    control_beta: float,
    variant_alpha: float,
    variant_beta: float,
    control_label: str = "Control",
    variant_label: str = "Variant",
) -> go.Figure:
    """Overlaid posterior Beta density curves for control and variant."""
    lo = min(sp_stats.beta.ppf(0.001, control_alpha, control_beta), sp_stats.beta.ppf(0.001, variant_alpha, variant_beta))
    hi = max(sp_stats.beta.ppf(0.999, control_alpha, control_beta), sp_stats.beta.ppf(0.999, variant_alpha, variant_beta))
    x = np.linspace(max(lo, 1e-6), min(hi, 1 - 1e-6), 500)

    fig = go.Figure()
    for label, a, b, color in (
        (control_label, control_alpha, control_beta, CONTROL_COLOR),
        (variant_label, variant_alpha, variant_beta, VARIANT_COLOR),
    ):
        y = sp_stats.beta.pdf(x, a, b)
        fig.add_trace(
            go.Scatter(
                x=x * 100,
                y=y,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, 0.15),
            )
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        title="Posterior Distributions of Conversion Rate",
        xaxis_title="Conversion rate (%)",
        yaxis_title="Posterior density",
    )
    _apply_axis_style(fig)
    return fig


def expected_loss_chart(loss_choose_control: float, loss_choose_variant: float) -> go.Figure:
    """Bar chart comparing the expected loss of shipping each arm."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Ship Control", "Ship Variant"],
            y=[loss_choose_control * 100, loss_choose_variant * 100],
            marker_color=[CONTROL_COLOR, VARIANT_COLOR],
            text=[f"{loss_choose_control*100:.3f} pp", f"{loss_choose_variant*100:.3f} pp"],
            textposition="outside",
            textfont=dict(color=INK_PRIMARY),
            width=0.5,
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Expected Loss by Decision (percentage points of conversion rate)",
        yaxis_title="Expected loss (pp)",
        showlegend=False,
        bargap=0.4,
    )
    _apply_axis_style(fig)
    return fig


def sample_size_curve_chart(
    baseline_rate: float, mde_range: np.ndarray, required_ns: np.ndarray, current_mde: float | None = None
) -> go.Figure:
    """Line chart of required per-group sample size across a range of MDEs."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mde_range * 100,
            y=required_ns,
            mode="lines",
            line=dict(color=CONTROL_COLOR, width=2),
            name="Required n per group",
        )
    )
    if current_mde is not None:
        fig.add_vline(x=current_mde * 100, line_width=1.5, line_dash="dash", line_color=INK_SECONDARY)
    fig.update_layout(
        **_BASE_LAYOUT,
        title=f"Required Sample Size vs. Minimum Detectable Effect (baseline = {baseline_rate*100:.1f}%)",
        xaxis_title="Relative minimum detectable effect (%)",
        yaxis_title="Required sample size per group",
        showlegend=False,
    )
    _apply_axis_style(fig)
    return fig


def sequential_monitoring_chart(days, z_stats, boundaries) -> go.Figure:
    """Line chart of the cumulative z-statistic vs. the O'Brien-Fleming
    stopping boundary over time (single axis: z-scale). The region beyond
    the boundary is shaded to visually flag 'safe to stop' territory."""
    days = list(days)
    z_stats = list(z_stats)
    boundaries = list(boundaries)
    upper = boundaries
    lower = [-b for b in boundaries]

    fig = go.Figure()
    # Shaded "significant" zones (critical status color, very low alpha)
    fig.add_trace(
        go.Scatter(x=days, y=upper, mode="lines", line=dict(color=STATUS_CRITICAL, width=1.5, dash="dash"),
                    name="Upper stopping boundary")
    )
    fig.add_trace(
        go.Scatter(x=days, y=lower, mode="lines", line=dict(color=STATUS_CRITICAL, width=1.5, dash="dash"),
                    name="Lower stopping boundary")
    )
    fig.add_trace(
        go.Scatter(x=days, y=z_stats, mode="lines+markers", line=dict(color=CONTROL_COLOR, width=2),
                    marker=dict(size=6), name="Observed z-statistic")
    )
    fig.add_hline(y=0, line_width=1, line_color=BASELINE)

    fig.update_layout(
        **_BASE_LAYOUT,
        title="Sequential Monitoring: Test Statistic vs. O'Brien-Fleming Boundary",
        xaxis_title="Day",
        yaxis_title="z-statistic",
    )
    _apply_axis_style(fig)
    return fig


def multivariant_bar_chart(control_label, control_rate, variant_labels, variant_rates, significant_flags) -> go.Figure:
    """Grouped bar chart of conversion rate per arm (control + variants),
    each in its fixed categorical color slot. Significant variants get a
    text annotation (icon + label, not color alone) above the bar."""
    labels = [control_label] + list(variant_labels)
    rates = [control_rate] + list(variant_rates)
    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(labels))]
    sig_flags = [False] + list(significant_flags)

    texts = []
    for r, sig in zip(rates, sig_flags):
        label = f"{r*100:.2f}%"
        if sig:
            label += "  ★ sig."
        texts.append(label)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[r * 100 for r in rates],
            marker_color=colors,
            text=texts,
            textposition="outside",
            textfont=dict(color=INK_PRIMARY),
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Conversion Rate by Variant (★ = significant vs. control after BH correction)",
        yaxis_title="Conversion rate (%)",
        showlegend=False,
        bargap=0.35,
    )
    _apply_axis_style(fig)
    return fig


def p_value_comparison_chart(variant_labels, raw_p, bonferroni_p, bh_p, alpha: float = 0.05) -> go.Figure:
    """Grouped bar chart comparing raw vs. Bonferroni vs. BH-adjusted p-values
    per variant, with the alpha threshold marked as a reference line."""
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Raw p-value", x=variant_labels, y=raw_p, marker_color=CATEGORICAL[0]))
    fig.add_trace(go.Bar(name="Bonferroni-adjusted", x=variant_labels, y=bonferroni_p, marker_color=CATEGORICAL[1]))
    fig.add_trace(go.Bar(name="Benjamini-Hochberg-adjusted", x=variant_labels, y=bh_p, marker_color=CATEGORICAL[2]))
    fig.add_hline(y=alpha, line_width=1.5, line_dash="dash", line_color=STATUS_CRITICAL,
                  annotation_text=f"alpha = {alpha}", annotation_font_color=INK_SECONDARY)
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Raw vs. Corrected p-values by Variant",
        yaxis_title="p-value",
        barmode="group",
    )
    _apply_axis_style(fig)
    return fig
