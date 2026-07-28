import pandas as pd

from analysis.cookie_cats_analysis import analyze_retention


def test_analyze_retention_wires_stats_correctly():
    df = pd.DataFrame(
        {
            "version": ["gate_30"] * 100 + ["gate_40"] * 100,
            "retention_1": [1] * 40 + [0] * 60 + [1] * 55 + [0] * 45,
        }
    )
    z, power, verdict = analyze_retention(df, "retention_1")

    assert z.control_n == 100
    assert z.variant_n == 100
    assert z.control_conversions == 40
    assert z.variant_conversions == 55
    assert verdict.decision in {"ship", "hold", "no-ship"}
    assert 0.0 <= power.power <= 1.0
