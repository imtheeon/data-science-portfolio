"""Bayesian A/B testing using the Beta-Binomial conjugate model.

Model
-----
Conversion is modeled as a Bernoulli process with unknown rate theta.
Using a Beta(alpha, beta) prior (default: Beta(1, 1), i.e. uniform / no
prior opinion) is conjugate to the Binomial likelihood, so after observing
`conversions` successes out of `n` trials the posterior is again a Beta
distribution:

    posterior = Beta(alpha_prior + conversions, beta_prior + n - conversions)

This module computes, for a control and a variant arm:
  * the posterior distributions themselves,
  * P(variant > control)   -- probability the variant is truly better,
  * expected loss for each decision (choose control / choose variant),
  * a credible interval for each arm's conversion rate.

P(variant > control) and expected loss do not have simple closed forms for
arbitrary Beta parameters, so they are estimated with Monte Carlo sampling
from the two posteriors (a standard, well-accepted technique in applied
Bayesian A/B testing). A fixed random seed keeps results reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

DEFAULT_PRIOR_ALPHA = 1.0
DEFAULT_PRIOR_BETA = 1.0
DEFAULT_N_SAMPLES = 200_000
DEFAULT_SEED = 42


@dataclass
class BetaPosterior:
    """Posterior Beta(alpha, beta) distribution for one arm's conversion rate."""

    alpha: float
    beta: float
    n: int
    conversions: int

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, cred_mass: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval, e.g. the 2.5th/97.5th percentiles
        of Beta(alpha, beta) for a 95% interval."""
        lower = (1 - cred_mass) / 2
        upper = 1 - lower
        return (
            float(sp_stats.beta.ppf(lower, self.alpha, self.beta)),
            float(sp_stats.beta.ppf(upper, self.alpha, self.beta)),
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return sp_stats.beta.pdf(x, self.alpha, self.beta)

    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.beta(self.alpha, self.beta, size=n_samples)


@dataclass
class BayesianResult:
    control_posterior: BetaPosterior
    variant_posterior: BetaPosterior
    prob_variant_beats_control: float
    expected_loss_choose_control: float  # expected regret if control is (wrongly) shipped
    expected_loss_choose_variant: float  # expected regret if variant is (wrongly) shipped
    control_ci: tuple[float, float]
    variant_ci: tuple[float, float]
    credible_mass: float


def beta_binomial_posterior(
    conversions: int,
    n: int,
    prior_alpha: float = DEFAULT_PRIOR_ALPHA,
    prior_beta: float = DEFAULT_PRIOR_BETA,
) -> BetaPosterior:
    """Compute the Beta posterior for one arm given observed data.

    posterior_alpha = prior_alpha + conversions
    posterior_beta  = prior_beta + (n - conversions)
    """
    if n < 0 or conversions < 0 or conversions > n:
        raise ValueError("Require 0 <= conversions <= n.")
    return BetaPosterior(
        alpha=prior_alpha + conversions,
        beta=prior_beta + (n - conversions),
        n=n,
        conversions=conversions,
    )


def bayesian_ab_test(
    control_conversions: int,
    control_n: int,
    variant_conversions: int,
    variant_n: int,
    prior_alpha: float = DEFAULT_PRIOR_ALPHA,
    prior_beta: float = DEFAULT_PRIOR_BETA,
    credible_mass: float = 0.95,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> BayesianResult:
    """Run a full Bayesian comparison of a control and a variant arm.

    P(variant > control) is estimated as the fraction of Monte Carlo
    posterior draws where theta_variant > theta_control.

    Expected loss uses the standard "regret" definition from Bayesian A/B
    testing (see e.g. Chris Stucchio's "Bayesian A/B Testing at VWO"):
        loss(choose control) = E[ max(theta_variant - theta_control, 0) ]
        loss(choose variant) = E[ max(theta_control - theta_variant, 0) ]
    i.e. the expected conversion-rate points left on the table if you pick
    the wrong arm, averaged over the joint posterior.
    """
    control_post = beta_binomial_posterior(control_conversions, control_n, prior_alpha, prior_beta)
    variant_post = beta_binomial_posterior(variant_conversions, variant_n, prior_alpha, prior_beta)

    rng = np.random.default_rng(seed)
    control_samples = rng.beta(control_post.alpha, control_post.beta, size=n_samples)
    variant_samples = rng.beta(variant_post.alpha, variant_post.beta, size=n_samples)

    prob_variant_wins = float(np.mean(variant_samples > control_samples))

    loss_choose_control = float(np.mean(np.maximum(variant_samples - control_samples, 0.0)))
    loss_choose_variant = float(np.mean(np.maximum(control_samples - variant_samples, 0.0)))

    return BayesianResult(
        control_posterior=control_post,
        variant_posterior=variant_post,
        prob_variant_beats_control=prob_variant_wins,
        expected_loss_choose_control=loss_choose_control,
        expected_loss_choose_variant=loss_choose_variant,
        control_ci=control_post.credible_interval(credible_mass),
        variant_ci=variant_post.credible_interval(credible_mass),
        credible_mass=credible_mass,
    )


def probability_each_variant_is_best(
    conversions: list[int],
    ns: list[int],
    prior_alpha: float = DEFAULT_PRIOR_ALPHA,
    prior_beta: float = DEFAULT_PRIOR_BETA,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Monte-Carlo P(arm i is the best arm) for an arbitrary number of arms.

    Draws `n_samples` posterior samples per arm, then for each draw records
    which arm had the highest sampled conversion rate. The proportion of
    draws each arm 'wins' approximates its probability of truly being best.
    """
    if len(conversions) != len(ns):
        raise ValueError("conversions and ns must be the same length.")
    rng = np.random.default_rng(seed)
    samples = np.column_stack(
        [
            rng.beta(prior_alpha + c, prior_beta + (n - c), size=n_samples)
            for c, n in zip(conversions, ns)
        ]
    )
    winners = np.argmax(samples, axis=1)
    counts = np.bincount(winners, minlength=len(conversions))
    return counts / n_samples
