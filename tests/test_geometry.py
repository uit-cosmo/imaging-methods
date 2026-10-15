"""calculate_splinted_LCFS on elliptic boundaries, in centimetres as C-Mod's EFIT has
them."""

import numpy as np
import pytest

from imaging_methods import calculate_splinted_LCFS

THETA = np.linspace(0, 2 * np.pi, 110, endpoint=False)  # 110 points, as C-Mod's files
EFIT_TIME = np.array([1.00, 1.02, 1.04])


def ellipse(r0, a=22.0, b=35.0):
    """``(R, Z)`` of an ellipse centred at (r0, 0), its R semi-axis ``a``, Z ``b``."""
    return r0 + a * np.cos(THETA), b * np.sin(THETA)


def outboard(r0, z, a=22.0, b=35.0):
    """The ellipse's outboard R at height ``z``."""
    return r0 + a * np.sqrt(1 - (z / b) ** 2)


def boundaries(*centres):
    """``(rbbbs, zbbbs)``, (point, efit_time): one ellipse per EFIT time."""
    rs, zs = zip(*(ellipse(r0) for r0 in centres))
    return np.stack(rs, axis=1), np.stack(zs, axis=1)


def test_the_outboard_leg_at_the_nearest_efit_time():
    rbbbs, zbbbs = boundaries(68.0, 69.0, 70.0)
    for t, r0 in ((0.99, 68.0), (1.012, 69.0), (1.026, 69.0), (1.5, 70.0)):
        r_fine, z_fine = calculate_splinted_LCFS(t, EFIT_TIME, rbbbs, zbbbs)
        np.testing.assert_array_equal(z_fine, np.linspace(-8, 1, 100))
        # The inboard leg, R < 86 cm, is left out of the spline: with it the heights
        # would repeat.
        np.testing.assert_allclose(r_fine, outboard(r0, z_fine), rtol=0, atol=1e-5)


def test_lists_are_taken_as_arrays():
    rbbbs, zbbbs = boundaries(68.0, 69.0, 70.0)
    expected = calculate_splinted_LCFS(1.02, EFIT_TIME, rbbbs, zbbbs)
    got = calculate_splinted_LCFS(1.02, EFIT_TIME, rbbbs.tolist(), zbbbs.tolist())
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


def test_a_leg_that_does_not_span_the_view_raises():
    r, z = ellipse(68.0, b=6.0)  # outboard of 86 cm only for |Z| < 3.5 cm
    with pytest.raises(ValueError):
        calculate_splinted_LCFS(1.0, EFIT_TIME[:1], r[:, None], z[:, None])
