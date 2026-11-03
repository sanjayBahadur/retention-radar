"""Small BG/NBD and Gamma-Gamma baselines for repeat shopping occasions.

Equations follow Fader, Hardie & Lee (2005), also documented in the lifetimes
BetaGeoFitter and GammaGammaFitter implementations. Times are measured in weeks.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln, hyp2f1


def fit_bgnbd_mle(
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit BG/NBD by likelihood; raise on failure instead of fabricating a fit."""
    x, last, age = (np.asarray(value, dtype=float) for value in (frequency, recency, T))
    if len(x) == 0:
        raise ValueError("BG/NBD needs purchase history.")

    def negative_likelihood(parameters):
        r, alpha, a, b = parameters
        base = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        base += betaln(a, b + x) - betaln(a, b)
        active = -(r + x) * np.log(alpha + age)
        inactive = np.log(a / (b + np.maximum(x, 1) - 1)) - (r + x) * np.log(alpha + last)
        likelihood = base + np.where(x > 0, np.logaddexp(active, inactive), active)
        return float(-likelihood.mean())

    fit = minimize(
        negative_likelihood,
        [1.0, 10.0, 1.5, 2.0],
        method="L-BFGS-B",
        bounds=[(1e-4, 100), (1e-4, 1000), (1.001, 100), (1e-4, 100)],
    )
    if not fit.success:
        raise RuntimeError(f"BG/NBD optimization failed: {fit.message}")
    return tuple(float(value) for value in fit.x)


def predict_bgnbd_purchases(
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
    params: tuple[float, float, float, float],
    future_t: float = 26.0,
) -> np.ndarray:
    """Conditional expected repeat purchases over future_t weeks (equation 10)."""
    if future_t < 0:
        raise ValueError("Forecast duration must be nonnegative.")
    r, alpha, a, b = params
    if a <= 1:
        raise ValueError("This baseline constrains a > 1 for a finite expectation.")
    x = np.asarray(frequency, dtype=float)
    age = np.asarray(T, dtype=float)
    z = future_t / (alpha + age + future_t)
    hypergeometric = hyp2f1(r + x, b + x, a + b + x - 1, z)
    remaining = 1 - np.exp(
        np.log(hypergeometric) + (r + x) * np.log((alpha + age) / (alpha + age + future_t))
    )
    expected = (a + b + x - 1) / (a - 1) * remaining
    expected *= predict_p_alive(frequency, recency, T, params)
    if not np.isfinite(expected).all():
        raise ValueError("BG/NBD produced a non-finite purchase forecast.")
    return np.maximum(expected, 0)


def predict_p_alive(
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
    params: tuple[float, float, float, float],
) -> np.ndarray:
    """Probability of remaining active under the BG/NBD assumptions."""
    r, alpha, a, b = params
    x, last, age = (np.asarray(value, dtype=float) for value in (frequency, recency, T))
    log_odds = np.log(a / (b + np.maximum(x, 1) - 1))
    log_odds += (r + x) * np.log((alpha + age) / (alpha + last))
    return np.where(x > 0, expit(-log_odds), 1.0)


def fit_gamma_gamma(
    frequency: np.ndarray,
    monetary_value: np.ndarray,
) -> tuple[float, float, float]:
    """Fit positive repeat-occasion spending, with q > 1 for a positive mean."""
    frequency = np.asarray(frequency, dtype=float)
    monetary_value = np.asarray(monetary_value, dtype=float)
    repeat = (frequency > 0) & (monetary_value > 0)
    x, spend = frequency[repeat], monetary_value[repeat]
    if len(x) == 0:
        raise ValueError("Gamma-Gamma needs positive repeat-occasion spending.")

    def negative_likelihood(parameters):
        p, q, v = parameters
        shape = p * x
        likelihood = gammaln(shape + q) - gammaln(shape) - gammaln(q)
        likelihood += q * np.log(v) + (shape - 1) * np.log(spend) + shape * np.log(x)
        likelihood -= (shape + q) * np.log(v + spend * x)
        return float(-likelihood.mean())

    fit = minimize(
        negative_likelihood,
        [2.0, 3.0, 15.0],
        method="L-BFGS-B",
        bounds=[(1e-3, 100), (1.001, 100), (1e-3, 1000)],
    )
    if not fit.success:
        raise RuntimeError(f"Gamma-Gamma optimization failed: {fit.message}")
    return tuple(float(value) for value in fit.x)


def predict_monetary_value(
    frequency: np.ndarray,
    monetary_value: np.ndarray,
    params: tuple[float, float, float],
) -> np.ndarray:
    """Shrink mean repeat-occasion spend toward the fitted population mean."""
    p, q, v = params
    x, spend = np.asarray(frequency, dtype=float), np.asarray(monetary_value, dtype=float)
    population_mean = p * v / (q - 1)
    weight = p * x / (p * x + q - 1)
    return weight * spend + (1 - weight) * population_mean
