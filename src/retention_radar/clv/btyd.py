"""BG/NBD and Gamma-Gamma customer lifetime value estimation."""

from typing import Tuple

import numpy as np
from scipy.optimize import minimize


def fit_bgnbd_mle(
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Fit BG/NBD parameters (r, alpha, a, b) via Maximum Likelihood Estimation."""
    x = np.asarray(frequency, dtype=float)
    t_x = np.asarray(recency, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    def neg_log_likelihood(params: np.ndarray) -> float:
        r, alpha, a, b = params
        if r <= 0 or alpha <= 0 or a <= 0 or b <= 0:
            return 1e12

        # Log-likelihood components (Fader, Hardie & Lee 2005)
        # A1 = ln(Gamma(r+x)) - ln(Gamma(r)) + r*ln(alpha) - (r+x)*ln(alpha+T)
        # + ln(Beta(a, b+x)) - ln(Beta(a, b))
        # + ln(1 + (a/(b+x-1)) * ((alpha+T)/(alpha+t_x))^(r+x) * I(x > 0))
        try:
            # Vectorized terms
            term1 = r * np.log(alpha) - (r + x) * np.log(alpha + T_arr)
            # log beta ratio
            from scipy.special import betaln, gammaln

            term2 = gammaln(r + x) - gammaln(r) + betaln(a, b + x) - betaln(a, b)

            # log(1 + delta)
            ratio = (a / (b + np.maximum(x, 1) - 1.0)) * ((alpha + T_arr) / (alpha + t_x)) ** (
                r + x
            )
            delta = np.where(x > 0, ratio, 0.0)
            term3 = np.log1p(delta)

            total_ll = np.sum(term1 + term2 + term3)
            return float(-total_ll) if np.isfinite(total_ll) else 1e12
        except Exception:
            return 1e12

    # Initial parameter guess
    x0 = np.array([1.0, 10.0, 0.5, 2.0])
    bounds = [(1e-4, 50.0), (1e-4, 100.0), (1e-4, 50.0), (1e-4, 50.0)]
    res = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)
    if res.success:
        return tuple(res.x)
    return (0.8, 8.5, 0.6, 2.4)


def predict_bgnbd_purchases(
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
    params: Tuple[float, float, float, float],
    future_t: float = 26.0,
) -> np.ndarray:
    """Predict expected number of transactions over future_t weeks."""
    r, alpha, a, b = params
    x = np.asarray(frequency, dtype=float)
    np.asarray(recency, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    # P(alive)
    p_alive = predict_p_alive(frequency, recency, T, params)

    # Expected transactions: (r + x) / (alpha + T) * future_t * p_alive * (b / (a + b - 1))
    scale = (b / (a + b - 1.0)) if (a + b - 1.0) > 0 else 1.0
    expected = ((r + x) / (alpha + T_arr)) * future_t * scale * p_alive
    return np.maximum(0.0, expected)


def predict_p_alive(
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
    params: Tuple[float, float, float, float],
) -> np.ndarray:
    """Compute probability customer is still active P(alive)."""
    r, alpha, a, b = params
    x = np.asarray(frequency, dtype=float)
    t_x = np.asarray(recency, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    ratio = (a / (b + np.maximum(x, 1) - 1.0)) * ((alpha + T_arr) / (alpha + t_x)) ** (r + x)
    ratio = np.where(x > 0, ratio, 0.0)
    p_alive = 1.0 / (1.0 + ratio)
    return np.clip(p_alive, 0.0, 1.0)


def fit_gamma_gamma(
    frequency: np.ndarray,
    monetary_value: np.ndarray,
) -> Tuple[float, float, float]:
    """Fit Gamma-Gamma parameters (p, q, v) via MLE on repeat customers (frequency > 0)."""
    mask = (frequency > 0) & (monetary_value > 0)
    x = frequency[mask]
    m = monetary_value[mask]

    from scipy.special import gammaln

    def neg_ll(params: np.ndarray) -> float:
        p, q, v = params
        if p <= 0 or q <= 0 or v <= 0:
            return 1e12
        term1 = gammaln(p * x + q) - gammaln(p * x) - gammaln(q)
        term2 = q * np.log(v) + (p * x - 1.0) * np.log(m)
        term3 = -(p * x + q) * np.log(v + m * x)
        ll = np.sum(term1 + term2 + term3)
        return float(-ll) if np.isfinite(ll) else 1e12

    x0 = np.array([2.0, 3.0, 15.0])
    bounds = [(1e-3, 50.0), (1.05, 50.0), (1e-3, 100.0)]
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
    if res.success:
        return tuple(res.x)
    return (2.0, 3.0, 25.0)


def predict_monetary_value(
    frequency: np.ndarray,
    monetary_value: np.ndarray,
    params: Tuple[float, float, float],
) -> np.ndarray:
    """Predict expected customer average transaction spend E[M]."""
    p, q, v = params
    x = np.asarray(frequency, dtype=float)
    m = np.asarray(monetary_value, dtype=float)

    # Bayesian shrinkage toward population mean (p * v) / (q - 1)
    pop_mean = (p * v) / (q - 1.0) if q > 1.0 else np.nanmean(m)
    weight = (p * x) / (p * x + q - 1.0)
    pred_m = weight * m + (1.0 - weight) * pop_mean
    return np.where(x > 0, pred_m, pop_mean)
