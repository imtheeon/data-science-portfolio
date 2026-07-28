from stats.frequentist import two_proportion_z_test
from stats.power import current_power
from analysis.verdict import recommend


def test_significant_positive_lift_ships():
    z = two_proportion_z_test(100, 10000, 400, 10000)  # huge, obvious win
    p = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n)
    verdict = recommend(z, p)
    assert verdict.decision == "ship"
    assert any("lifts conversion" in r for r in verdict.reasons)


def test_significant_negative_lift_does_not_ship():
    z = two_proportion_z_test(300, 2000, 255, 2000)  # variant significantly worse
    p = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n)
    verdict = recommend(z, p)
    assert verdict.decision == "no-ship"
    assert any("WORSE" in r for r in verdict.reasons)


def test_not_significant_low_power_holds():
    z = two_proportion_z_test(45, 300, 51, 300)  # small sample, inconclusive
    p = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n)
    verdict = recommend(z, p, min_power=0.95)  # force power below threshold
    assert verdict.decision == "hold"
    assert any("underpowered" in r for r in verdict.reasons)


def test_not_significant_adequate_power_no_ships():
    z = two_proportion_z_test(300, 3000, 300, 3000)  # identical rates, huge n
    p = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n)
    verdict = recommend(z, p, min_power=0.05)
    assert verdict.decision == "no-ship"
