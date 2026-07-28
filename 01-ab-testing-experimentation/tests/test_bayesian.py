"""Tests for stats/bayesian.py (Beta-Binomial Bayesian A/B model)."""

import numpy as np
import pytest
from scipy import stats as sp_stats

from stats.bayesian import (
    bayesian_ab_test,
    beta_binomial_posterior,
    probability_each_variant_is_best,
)


class TestBetaBinomialPosterior:
    def test_posterior_parameters_uniform_prior(self):
        post = beta_binomial_posterior(conversions=40, n=200, prior_alpha=1, prior_beta=1)
        assert post.alpha == pytest.approx(41)
        assert post.beta == pytest.approx(161)

    def test_posterior_mean_matches_scipy_beta_mean(self):
        post = beta_binomial_posterior(conversions=40, n=200)
        scipy_mean = sp_stats.beta.mean(post.alpha, post.beta)
        assert post.mean == pytest.approx(scipy_mean, abs=1e-10)

    def test_credible_interval_matches_scipy_ppf(self):
        post = beta_binomial_posterior(conversions=300, n=1000)
        lo, hi = post.credible_interval(0.95)
        assert lo == pytest.approx(sp_stats.beta.ppf(0.025, post.alpha, post.beta))
        assert hi == pytest.approx(sp_stats.beta.ppf(0.975, post.alpha, post.beta))
        assert lo < post.mean < hi

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            beta_binomial_posterior(conversions=10, n=5)


class TestBayesianABTest:
    def test_obvious_winner_high_probability(self):
        # Variant clearly better: 15% vs 10% on large samples
        result = bayesian_ab_test(
            control_conversions=1000, control_n=10000,
            variant_conversions=1500, variant_n=10000,
        )
        assert result.prob_variant_beats_control > 0.99
        # Shipping the variant is (almost) risk-free here: expected loss from
        # choosing variant should be tiny since it is essentially always better.
        assert result.expected_loss_choose_variant < 0.001
        # Loss of wrongly sticking with control should be near the true
        # observed ~0.05 gap, since variant wins in nearly every posterior draw.
        assert result.expected_loss_choose_control == pytest.approx(0.05, abs=0.01)
        assert result.expected_loss_choose_control > result.expected_loss_choose_variant

    def test_no_real_difference_prob_near_half(self):
        result = bayesian_ab_test(
            control_conversions=500, control_n=5000,
            variant_conversions=505, variant_n=5000,
        )
        assert 0.3 < result.prob_variant_beats_control < 0.9

    def test_credible_intervals_are_ordered(self):
        result = bayesian_ab_test(
            control_conversions=200, control_n=2000,
            variant_conversions=260, variant_n=2000,
        )
        assert result.control_ci[0] < result.control_ci[1]
        assert result.variant_ci[0] < result.variant_ci[1]

    def test_reproducible_with_fixed_seed(self):
        r1 = bayesian_ab_test(100, 1000, 130, 1000, seed=99)
        r2 = bayesian_ab_test(100, 1000, 130, 1000, seed=99)
        assert r1.prob_variant_beats_control == r2.prob_variant_beats_control


class TestProbabilityEachVariantIsBest:
    def test_probabilities_sum_to_one(self):
        probs = probability_each_variant_is_best(
            conversions=[100, 120, 90], ns=[1000, 1000, 1000]
        )
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)

    def test_best_arm_has_highest_probability(self):
        probs = probability_each_variant_is_best(
            conversions=[100, 200, 90], ns=[1000, 1000, 1000]
        )
        assert probs[1] == max(probs)

    def test_symmetric_arms_roughly_equal(self):
        probs = probability_each_variant_is_best(
            conversions=[100, 100], ns=[1000, 1000], seed=123
        )
        assert probs[0] == pytest.approx(0.5, abs=0.05)
        assert probs[1] == pytest.approx(0.5, abs=0.05)
