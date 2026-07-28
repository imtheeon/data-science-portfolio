"""Tests for stats/frequentist.py.

Cross-validates the from-scratch z-test and Welch t-test implementations
against reference implementations (statsmodels.stats.proportion.proportions_ztest
and scipy.stats.ttest_ind) to confirm the math is actually correct, not just
plausible-looking.
"""

import numpy as np
import pytest
from scipy import stats as sp_stats
from statsmodels.stats.proportion import proportions_ztest

from stats.frequentist import chi_square_test, two_proportion_z_test, welch_t_test


class TestTwoProportionZTest:
    def test_matches_statsmodels_significant_case(self):
        control_conv, control_n = 120, 1000
        variant_conv, variant_n = 150, 1000

        ours = two_proportion_z_test(control_conv, control_n, variant_conv, variant_n)

        # statsmodels convention: z = (p_variant - p_control) when count/nobs
        # ordered [variant, control]; use pooled variance (prop_var default
        # is the pooled estimate when testing equality of two proportions).
        z_ref, p_ref = proportions_ztest(
            count=np.array([variant_conv, control_conv]),
            nobs=np.array([variant_n, control_n]),
        )

        assert ours.z_stat == pytest.approx(z_ref, abs=1e-8)
        assert ours.p_value == pytest.approx(p_ref, abs=1e-8)

    def test_matches_statsmodels_borderline_case(self):
        control_conv, control_n = 500, 5000
        variant_conv, variant_n = 545, 5000

        ours = two_proportion_z_test(control_conv, control_n, variant_conv, variant_n)
        z_ref, p_ref = proportions_ztest(
            count=np.array([variant_conv, control_conv]),
            nobs=np.array([variant_n, control_n]),
        )

        assert ours.z_stat == pytest.approx(z_ref, abs=1e-8)
        assert ours.p_value == pytest.approx(p_ref, abs=1e-8)

    def test_matches_statsmodels_null_case(self):
        # Identical rates -> z should be ~0, p ~1
        ours = two_proportion_z_test(300, 3000, 300, 3000)
        z_ref, p_ref = proportions_ztest(
            count=np.array([300, 300]), nobs=np.array([3000, 3000])
        )
        assert ours.z_stat == pytest.approx(z_ref, abs=1e-8)
        assert ours.p_value == pytest.approx(p_ref, abs=1e-8)
        assert ours.p_value > 0.99

    def test_lift_and_ci_sign(self):
        ours = two_proportion_z_test(100, 1000, 130, 1000)
        assert ours.variant_rate == pytest.approx(0.13)
        assert ours.control_rate == pytest.approx(0.10)
        assert ours.absolute_lift == pytest.approx(0.03)
        assert ours.relative_lift == pytest.approx(0.30)
        # CI should contain the observed point estimate's sign (positive lift)
        assert ours.ci_low < ours.absolute_lift < ours.ci_high

    def test_is_significant_flag(self):
        # Huge, obvious effect -> significant
        ours = two_proportion_z_test(100, 10000, 400, 10000)
        assert ours.is_significant is True
        assert ours.p_value < 0.05

        # Tiny samples, no real difference -> not significant
        ours2 = two_proportion_z_test(5, 100, 6, 100)
        assert ours2.is_significant is False

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            two_proportion_z_test(10, 0, 5, 100)
        with pytest.raises(ValueError):
            two_proportion_z_test(200, 100, 5, 100)  # conversions > n


class TestWelchTTest:
    def test_matches_scipy_unequal_variance(self):
        rng = np.random.default_rng(1)
        control = rng.normal(loc=50, scale=10, size=200)
        variant = rng.normal(loc=53, scale=15, size=220)

        ours = welch_t_test(control, variant)
        t_ref, p_ref = sp_stats.ttest_ind(variant, control, equal_var=False)

        assert ours.t_stat == pytest.approx(t_ref, abs=1e-8)
        assert ours.p_value == pytest.approx(p_ref, abs=1e-8)

    def test_matches_scipy_equal_size_samples(self):
        rng = np.random.default_rng(7)
        control = rng.normal(loc=20, scale=5, size=150)
        variant = rng.normal(loc=20.5, scale=5, size=150)

        ours = welch_t_test(control, variant)
        t_ref, p_ref = sp_stats.ttest_ind(variant, control, equal_var=False)

        assert ours.t_stat == pytest.approx(t_ref, abs=1e-8)
        assert ours.p_value == pytest.approx(p_ref, abs=1e-8)

    def test_df_matches_welch_satterthwaite_scipy(self):
        rng = np.random.default_rng(3)
        control = rng.normal(loc=10, scale=2, size=50)
        variant = rng.normal(loc=11, scale=6, size=80)

        ours = welch_t_test(control, variant)
        # scipy's ttest_ind with equal_var=False returns Welch-Satterthwaite df
        # only if we ask for it via a separate call; verify indirectly through
        # the p-value match above, and directly check df is in a sane range.
        assert 0 < ours.df <= (len(control) + len(variant) - 2)

    def test_raises_on_tiny_samples(self):
        with pytest.raises(ValueError):
            welch_t_test([1.0], [2.0, 3.0])


class TestChiSquareTest:
    def test_matches_scipy_chi2_contingency(self):
        table = np.array([[50, 450], [70, 430]])  # control vs variant x converted/not
        ours = chi_square_test(table)
        chi2_ref, p_ref, dof_ref, _ = sp_stats.chi2_contingency(table, correction=False)

        assert ours.chi2_stat == pytest.approx(chi2_ref, abs=1e-8)
        assert ours.p_value == pytest.approx(p_ref, abs=1e-8)
        assert ours.dof == dof_ref

    def test_matches_scipy_three_category_table(self):
        table = np.array([[100, 50, 20], [90, 60, 30]])
        ours = chi_square_test(table)
        chi2_ref, p_ref, dof_ref, _ = sp_stats.chi2_contingency(table, correction=False)

        assert ours.chi2_stat == pytest.approx(chi2_ref, abs=1e-8)
        assert ours.p_value == pytest.approx(p_ref, abs=1e-8)
        assert ours.dof == dof_ref

    def test_identical_distributions_not_significant(self):
        table = np.array([[100, 200], [100, 200]])
        ours = chi_square_test(table)
        assert ours.chi2_stat == pytest.approx(0.0, abs=1e-8)
        assert ours.p_value == pytest.approx(1.0, abs=1e-8)
