"""Real-data A/B test analysis: Cookie Cats gate placement.

gate_30 (control) vs. gate_40 (variant) — does moving the level-30 paywall
gate to level 40 change 1-day and 7-day retention? Uses the real Kaggle
dataset loaded by data.load_cookie_cats, and the same from-scratch
two-proportion z-test / power analysis used throughout this project.
"""

from __future__ import annotations

import pandas as pd

from analysis.verdict import ShipVerdict, recommend
from data.load_cookie_cats import download
from stats.frequentist import ZTestResult, two_proportion_z_test
from stats.power import PowerResult, current_power


def load_data() -> pd.DataFrame:
    return pd.read_csv(download())


def analyze_retention(
    df: pd.DataFrame, retention_col: str, alpha: float = 0.05
) -> tuple[ZTestResult, PowerResult, ShipVerdict]:
    control = df[df["version"] == "gate_30"]
    variant = df[df["version"] == "gate_40"]

    control_n = len(control)
    variant_n = len(variant)
    control_conv = int(control[retention_col].sum())
    variant_conv = int(variant[retention_col].sum())

    z_result = two_proportion_z_test(
        control_conv, control_n, variant_conv, variant_n, alpha=alpha
    )
    power_result = current_power(
        z_result.control_rate, z_result.variant_rate, control_n, variant_n, alpha=alpha
    )
    verdict = recommend(z_result, power_result)
    return z_result, power_result, verdict


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} real players from the Cookie Cats dataset.")
    for col in ("retention_1", "retention_7"):
        z, p, v = analyze_retention(df, col)
        print(f"\n=== {col} ===")
        print(f"Control (gate_30) rate: {z.control_rate:.4f} (n={z.control_n})")
        print(f"Variant (gate_40) rate: {z.variant_rate:.4f} (n={z.variant_n})")
        print(f"p-value: {z.p_value:.4f}  |  95% CI on lift: [{z.ci_low:.4f}, {z.ci_high:.4f}]")
        print(f"Achieved power: {p.power:.4f}")
        print(f"Verdict: {v.decision.upper()}")
        for reason in v.reasons:
            print(f"  - {reason}")
