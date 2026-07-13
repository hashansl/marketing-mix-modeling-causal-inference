"""
Media transformations for Marketing Mix Modeling.

These are the two functional-form assumptions at the heart of MMM:

    adstock     - how a dollar of spend carries over into future weeks
                  (a temporal-shape assumption on the treatment effect)
    saturation  - how the marginal effect of spend shrinks as spend grows
                  (a dose-response / effect-heterogeneity assumption)

This module is the *reference* NumPy implementation. Model 1 (frequentist
benchmark) uses it directly.
"""

import numpy as np


def geometric_adstock(x, theta, L=12, normalize=True):
    """
    Geometric (fixed-decay) adstock via a causal weighted convolution.

    The adstocked value at week t is a weighted sum of current and past spend,
    with weights theta**0, theta**1, ..., theta**L:

        a_t = sum_{l=0}^{L} theta**l * x_{t-l}   (optionally normalized)

    Implemented as a convolution rather than a recursive loop so it stays
    vectorized and differentiable - the same structure ports cleanly to
    PyTensor later.

    Parameters
    ----------
    x : array-like
        Spend series for a single channel (length T).
    theta : float in [0, 1)
        Decay/retention rate. Higher = longer carryover.
        (e.g. TV ~0.6-0.7, paid search ~0.1-0.2.)
    L : int
        Maximum lag (carryover window) in weeks.
    normalize : bool
        If True, weights sum to 1 so the adstocked series keeps the same
        scale as the input (a weighted moving average). Recommended.

    Returns
    -------
    np.ndarray of length T : the adstocked series.
    """
    x = np.asarray(x, dtype=float)
    if not (0.0 <= theta < 1.0):
        raise ValueError(f"theta must be in [0, 1), got {theta}")

    weights = theta ** np.arange(L + 1)
    if normalize:
        weights = weights / weights.sum()

    # np.convolve(x, weights)[t] = sum_l x[t-l] * weights[l], which is causal:
    # the value at t depends only on x at t and earlier. Trim to length T.
    
    # Convolution slides the weight kernel across x and, at each position, computes a sum of products. 
    
    return np.convolve(x, weights)[: len(x)]


def hill_saturation(x, alpha, kappa):
    """
    Hill (sigmoidal) saturation. Bounded in [0, 1), strictly increasing.

        s(x) = x**alpha / (kappa**alpha + x**alpha)

    kappa is the half-saturation point: s(kappa) = 0.5, which makes it
    interpretable ("the spend level at which we reach half of max effect").
    alpha controls the steepness of the S-curve.

    Parameters
    ----------
    x : array-like
        (Adstocked) spend, non-negative.
    alpha : float > 0
        Shape/steepness.
    kappa : float > 0
        Half-saturation point, on the same scale as x.

    Returns
    -------
    np.ndarray : saturated response in [0, 1).
    """
    x = np.asarray(x, dtype=float)
    if alpha <= 0 or kappa <= 0:
        raise ValueError("alpha and kappa must be positive")
    # guard against 0**negative etc.; x is expected non-negative
    xa = np.power(np.clip(x, 0, None), alpha)
    return xa / (kappa ** alpha + xa)


def logistic_saturation(x, lam):
    """
    Logistic saturation - the alternative form used in the functional-form
    sensitivity analysis (swap this in for hill_saturation and watch ROI move).

        s(x) = (1 - exp(-lam * x)) / (1 + exp(-lam * x))

    Parameters
    ----------
    x : array-like
        (Adstocked) spend, non-negative.
    lam : float > 0
        Saturation rate.

    Returns
    -------
    np.ndarray : saturated response in [0, 1).
    """
    x = np.asarray(x, dtype=float)
    if lam <= 0:
        raise ValueError("lam must be positive")
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))

def delayed_adstock(x, alpha, theta, L, normalize=False):
    weights = np.array([alpha**(l-theta)**2 for l in range(L)])
    if normalize:
        weights = weights / weights.sum()
    adstocked_x = np.convolve(x, weights)[:len(x)]
    return adstocked_x