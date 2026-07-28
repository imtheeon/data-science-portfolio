"""Turns a hypothesis-test result and a power result into a plain-language
ship / hold / no-ship recommendation with the reasoning spelled out.

Decision rules
--------------
- Significant, positive lift            -> "ship"
- Significant, negative lift            -> "no-ship" (variant is harmful)
- Not significant, power < min_power    -> "hold" (test is underpowered;
                                            more data needed before concluding
                                            "no effect")
- Not significant, power >= min_power   -> "no-ship" (adequately powered
                                            test found no real effect)
"""

from __future__ import annotations

from dataclasses import dataclass

from stats.frequentist import ZTestResult
from stats.power import PowerResult


@dataclass
class ShipVerdict:
    decision: str  # "ship" | "hold" | "no-ship"
    reasons: list[str]


def recommend(
    z_result: ZTestResult,
    power_result: PowerResult,
    min_power: float = 0.8,
) -> ShipVerdict:
    reasons: list[str] = []

    if not z_result.is_significant:
        reasons.append(
            f"p-value {z_result.p_value:.4f} is not below alpha "
            f"{z_result.alpha} — result is not statistically significant."
        )
        if power_result.power < min_power:
            reasons.append(
                f"Achieved power is only {power_result.power:.2f} (below "
                f"{min_power}) — the test may simply be underpowered to "
                "detect the observed effect; collect more data before "
                "concluding there is no effect."
            )
            return ShipVerdict(decision="hold", reasons=reasons)
        reasons.append(
            f"Achieved power is {power_result.power:.2f}, so the test was "
            "adequately powered and still found no significant effect."
        )
        return ShipVerdict(decision="no-ship", reasons=reasons)

    if z_result.absolute_lift > 0:
        reasons.append(
            f"p-value {z_result.p_value:.4f} < alpha {z_result.alpha}; "
            f"variant lifts conversion rate by {z_result.absolute_lift * 100:.2f} "
            f"points (95% CI [{z_result.ci_low * 100:.2f}, "
            f"{z_result.ci_high * 100:.2f}])."
        )
        reasons.append(f"Achieved power: {power_result.power:.2f}.")
        return ShipVerdict(decision="ship", reasons=reasons)

    reasons.append(
        f"p-value {z_result.p_value:.4f} < alpha {z_result.alpha}; variant "
        f"is significantly WORSE than control by "
        f"{abs(z_result.absolute_lift) * 100:.2f} points."
    )
    return ShipVerdict(decision="no-ship", reasons=reasons)
