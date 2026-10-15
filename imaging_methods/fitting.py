"""Exponential fits to a profile: a two-sided exponential centred at zero, and a single
exponential.

Moved from fusion_scripts `waveform_analysis/fitting.py` at 12b8543.
"""

import numpy as np
from scipy.optimize import minimize

__all__ = ["fit_two_sided", "fit_exp"]


def _two_sided_initial_guess(coords, values, eps):
    """Data-driven start ``[scale, sigma, offset]`` for ``fit_two_sided``.

    The waveform peaks near ``x=0`` and decays to ``offset`` with independent
    left/right ``1/e`` lengths ``lam_left = scale*(1-sigma)`` and
    ``lam_right = scale*sigma``. We read those three quantities off the data and
    map them back to ``(scale, sigma, offset)``.
    """
    offset0 = float(np.min(values))
    peak = int(np.argmax(values))
    x_peak = float(coords[peak])
    amplitude = float(values[peak]) - offset0

    span = float(coords[-1] - coords[0])
    floor = span / len(coords)  # smallest length we can resolve
    if amplitude <= 0:  # degenerate (flat / inverted) data: fall back
        return [span / 4, 0.5, offset0]

    target = offset0 + amplitude / np.e

    def decay_length(indices):
        """Distance from the peak to the ``1/e`` crossing (interpolated)."""
        prev = peak
        for i in indices:
            if values[i] < target:
                v0, v1 = values[prev], values[i]
                frac = (v0 - target) / (v0 - v1) if v0 != v1 else 0.0
                x_cross = coords[prev] + frac * (coords[i] - coords[prev])
                return abs(x_cross - x_peak)
            prev = i
        return abs(coords[prev] - x_peak)  # no crossing: use the far edge

    lam_left = max(decay_length(range(peak - 1, -1, -1)), floor)
    lam_right = max(decay_length(range(peak + 1, len(coords))), floor)

    scale0 = lam_left + lam_right
    sigma0 = float(np.clip(lam_right / scale0, eps, 1 - eps))
    return [scale0, sigma0, float(np.clip(offset0, -10, 10))]


def fit_two_sided(coords, values):
    """
    Fit an asymmetric two-sided exponential profile centred at 0 to (coords, values).

    Model (parameters scale, sigma, offset):
        lam_left  = scale * (1 - sigma)
        lam_right = scale * sigma
        left  (x < 0):  offset + (1 - offset) * exp( x / lam_left )
        right (x >= 0): offset + (1 - offset) * exp(-x / lam_right)

    The parameters are found by minimising the sum of squared residuals with
    ``scipy.optimize.minimize`` (L-BFGS-B, box-constrained).

    Returns ``(popt, y_fit)`` where ``popt = [scale, sigma, offset]``.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)

    def model(x, scale, sigma, offset):
        lam_left = scale * (1 - sigma)
        lam_right = scale * sigma
        lam = np.where(x < 0, lam_left, lam_right)
        return offset + (1 - offset) * np.exp(-np.abs(x) / lam)

    def sum_squared_residuals(params):
        return np.sum((model(coords, *params) - values) ** 2)

    # Keep scale > 0 and sigma in (0, 1) strictly so lam_left/lam_right never vanish.
    eps = 1e-6
    p0 = _two_sided_initial_guess(coords, values, eps)
    bounds = [(eps, None), (eps, 1 - eps), (-10, 10)]

    result = minimize(
        sum_squared_residuals,
        p0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 10_000},
    )
    popt = result.x
    y_fit = model(coords, *popt)
    return popt, y_fit


def fit_exp(coords, values):
    """
    Fit an exponential profile to (coords, values).

    The model ``prefactor * exp(x / scale)`` is linear in log space,
    ``log(y) = log(prefactor) + x / scale``, so the parameters are obtained from
    a straight-line least-squares fit ``log(values) ~ intercept + slope * x``
    with ``scale = 1 / slope`` and ``prefactor = exp(intercept)``. Non-positive
    values have no logarithm and are excluded from the fit.

    Returns (popt, y_fit) where popt = [scale, prefactor].
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)

    positive = values > 0
    slope, intercept = np.polyfit(coords[positive], np.log(values[positive]), 1)

    scale = 1 / slope
    prefactor = np.exp(intercept)
    popt = np.array([scale, prefactor])
    y_fit = prefactor * np.exp(coords / scale)
    return popt, y_fit
