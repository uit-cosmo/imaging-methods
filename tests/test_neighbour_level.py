"""The contour level read off a conditional average: neighbour_level and its helpers."""

import numpy as np
import pytest
import xarray as xr

from imaging_methods import neighbour_level
from imaging_methods.contours import neighbour_ratios, zero_lag_frame

LAGS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def average_with(frame, lags=LAGS):
    """A conditional average over (y, x, time), as find_events_and_2dca returns it,
    whose frame at lag 0 is ``frame`` and whose other lags are a fraction of it."""
    frame = np.asarray(frame, dtype=float)
    weights = np.exp(-(lags**2))
    data = frame[..., None] * weights
    return xr.Dataset({"cond_av": (("y", "x", "time"), data)}, coords={"time": lags})


def peaked_frame(peak=(2, 3), shape=(5, 6), up=0.4, down=0.5, left=0.6, right=0.7):
    """A frame with 1 at ``peak`` (y, x) and the given values at its four
    neighbours."""
    frame = np.full(shape, 0.1)
    y, x = peak
    frame[y, x] = 1.0
    for (dy, dx), value in zip(
        ((-1, 0), (1, 0), (0, -1), (0, 1)), (up, down, left, right)
    ):
        if 0 <= y + dy < shape[0] and 0 <= x + dx < shape[1]:
            frame[y + dy, x + dx] = value
    return frame


def test_the_level_is_the_mean_of_the_four_neighbours_over_the_peak():
    average_ds = average_with(2.0 * peaked_frame())  # the level is a ratio
    assert neighbour_level(average_ds) == pytest.approx(0.55, rel=1e-12)
    assert type(neighbour_level(average_ds)) is float


def test_a_peak_on_the_border_averages_the_neighbours_it_has():
    frame = peaked_frame(peak=(0, 3), down=0.5, left=0.6, right=0.8)
    assert neighbour_level(average_with(frame)) == pytest.approx((0.5 + 0.6 + 0.8) / 3)
    assert neighbour_ratios(frame) == pytest.approx([0.5, 0.6, 0.8])


def test_step_reads_the_neighbours_further_out():
    frame = peaked_frame()
    frame[0, 3], frame[4, 3], frame[2, 1], frame[2, 5] = 0.2, 0.3, 0.25, 0.35
    assert neighbour_level(average_with(frame), step=2) == pytest.approx(0.275)
    assert neighbour_ratios(frame, step=2) == pytest.approx([0.2, 0.3, 0.25, 0.35])


def test_the_frame_is_the_one_at_the_lag_closest_to_zero():
    frame = peaked_frame()
    lags = np.array([-1.6, -0.6, 0.4, 1.4])
    average_ds = average_with(frame, lags)
    expected = frame * np.exp(-(0.4**2))
    np.testing.assert_array_equal(zero_lag_frame(average_ds), expected)
    # The dims of the variable are read by name.
    swapped = average_ds.transpose("time", "x", "y")
    np.testing.assert_array_equal(zero_lag_frame(swapped), expected)


def test_another_variable(fresh):
    """On the cross-correlation of a real 2DCA average (the decorrelation fixture's)."""
    average_ds = fresh[(2, 1)]
    level = neighbour_level(average_ds, "cross_corr")
    ratios = neighbour_ratios(zero_lag_frame(average_ds, "cross_corr"))
    assert level == float(ratios.mean()) and 0 < level < 1
