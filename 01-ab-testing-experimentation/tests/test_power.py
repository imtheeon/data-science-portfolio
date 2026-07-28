"""Tests for stats/power.py (sample size, power, days-to-significance)."""

import math

import pytest
from scipy import stats as sp_stats

from stats.power import current_power, days_to_significance, required_sample_size


class TestRequiredSampleSize:
    def test_matches_manual_formula(self):
        baseline = 0.10
        mde = 0.10  # relative
        alpha = 0.05
        power = 0.8

        result = required_sample_size(baseline, mde, alpha=alpha, power=power, relative=True)

        target = baseline * 1.10
        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        z_beta = sp_stats.norm.ppf(power)
        expected_n = (
            (z_alpha + z_beta) ** 2
            * (baseline * (1 - baseline) + target * (1 - target))
            / (target - baseline) ** 2
        )
        assert result.n_per_group == math.ceil(expected_n)

    def test_larger_mde_needs_smaller_sample(self):
        small_effect = required_sample_size(0.10, mde=0.05, relative=True)
        large_effect = required_sample_size(0.10, mde=0.30, relative=True)
        assert large_effect.n_per_group < small_effect.n_per_group

    def test_higher_power_needs_larger_sample(self):
        low_power = required_sample_size(0.10, mde=0.10, power=0.7)
        high_power = required_sample_size(0.10, mde=0.10, power=0.95)
        assert high_power.n_per_group > low_power.n_per_group

    def test_absolute_mde_mode(self):
        result = required_sample_size(0.10, mde=0.02, relative=False)
        assert result.target_rate == pytest.approx(0.12)
        assert result.mde_absolute == pytest.approx(0.02)

    def test_invalid_baseline_raises(self):
        with pytest.raises(ValueError):
            required_sample_size(0.0, mde=0.1)
        with pytest.raises(ValueError):
            required_sample_size(1.5, mde=0.1)


class TestCurrentPower:
    def test_matches_manual_formula(self):
        p1, p2, n1, n2, alpha = 0.10, 0.12, 5000, 5000, 0.05
        result = current_power(p1, p2, n1, n2, alpha=alpha)

        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        delta = abs(p2 - p1)
        expected_power = sp_stats.norm.cdf(delta / se - z_alpha) + sp_stats.norm.cdf(
            -delta / se - z_alpha
        )
        assert result.power == pytest.approx(expected_power, abs=1e-10)

    def test_power_increases_with_sample_size(self):
        small = current_power(0.10, 0.12, 500, 500)
        large = current_power(0.10, 0.12, 50000, 50000)
        assert large.power > small.power

    def test_zero_effect_low_power(self):
        result = current_power(0.10, 0.10, 1000, 1000)
        assert result.power < 0.10

    def test_large_effect_large_n_high_power(self):
        result = current_power(0.10, 0.20, 10000, 10000)
        assert result.power > 0.99


class TestDaysToSignificance:
    def test_computes_remaining_days(self):
        result = days_to_significance(
            n_per_group_required=10000, n_per_group_current=4000, daily_traffic_per_group=500
        )
        assert result.days_remaining == math.ceil((10000 - 4000) / 500)
        assert result.already_reached is False

    def test_already_reached_when_current_exceeds_required(self):
        result = days_to_significance(
            n_per_group_required=1000, n_per_group_current=5000, daily_traffic_per_group=200
        )
        assert result.already_reached is True
        assert result.days_remaining == 0

    def test_invalid_traffic_raises(self):
        with pytest.raises(ValueError):
            days_to_significance(1000, 100, 0)
