"""fit_two_sided and fit_exp on noiseless profiles whose parameters are known."""

import numpy as np
import pytest

from imaging_methods import fit_exp, fit_two_sided

X = np.linspace(-5, 5, 41)


def two_sided(x, scale, sigma, offset):
    lam = np.where(x < 0, scale * (1 - sigma), scale * sigma)
    return offset + (1 - offset) * np.exp(-np.abs(x) / lam)


@pytest.mark.parametrize(
    "scale, sigma, offset", [(3.0, 0.3, 0.1), (2.0, 0.5, 0.0), (4.0, 0.8, -0.2)]
)
def test_fit_two_sided_recovers_the_profile(scale, sigma, offset):
    values = two_sided(X, scale, sigma, offset)
    popt, y_fit = fit_two_sided(X, values)
    assert popt == pytest.approx([scale, sigma, offset], abs=1e-4)
    assert y_fit.shape == X.shape
    np.testing.assert_allclose(y_fit, values, rtol=0, atol=1e-5)


def test_fit_two_sided_on_a_flat_profile_stays_at_its_start():
    """No peak to read decay lengths off: the start is a quarter of the span, symmetric,
    at the profile's level, which already fits exactly."""
    popt, y_fit = fit_two_sided(X, np.ones_like(X))
    np.testing.assert_array_equal(popt, [2.5, 0.5, 1.0])
    np.testing.assert_array_equal(y_fit, np.ones_like(X))


def test_fit_exp_recovers_the_profile():
    values = 2.0 * np.exp(X / -1.5)
    popt, y_fit = fit_exp(X, values)
    assert popt == pytest.approx([-1.5, 2.0], rel=1e-12)
    np.testing.assert_allclose(y_fit, values, rtol=1e-12)


def test_fit_exp_leaves_out_what_has_no_logarithm():
    values = 2.0 * np.exp(X / -1.5)
    values[[3, 10]] = 0.0, -1.0
    popt, y_fit = fit_exp(X, values)
    assert popt == pytest.approx([-1.5, 2.0], rel=1e-12)
    np.testing.assert_allclose(y_fit, 2.0 * np.exp(X / -1.5), rtol=1e-12)
