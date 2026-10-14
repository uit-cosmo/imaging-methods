"""get_maximum_time and its helpers on Gaussian pulses whose peak lags are known."""

import warnings

import numpy as np
import pytest
import xarray as xr

from imaging_methods import get_maximum_time
from imaging_methods.maximum_trajectory import (
    find_maximum_interpolate,
    gaussian_convolve,
)

LAGS = np.linspace(-15e-6, 15e-6, 61)  # 0.5 us sampling, as an APD average's lags
NY, NX = 3, 4
# The lag each pixel peaks at: a pulse moving along x and, slower, along y.
PEAKS = np.array(
    [[(x - 1.5) * 1e-6 + (y - 1) * 0.5e-6 for x in range(NX)] for y in range(NY)]
)


def event(peaks=PEAKS, width=4e-6):
    """A conditional average's cond_av: Gaussian pulses over (y, x, time), with R and the
    reference pixel among its coordinates."""
    data = np.exp(-(((LAGS - peaks[..., None]) / width) ** 2))
    return xr.DataArray(
        data,
        dims=("y", "x", "time"),
        coords={
            "time": LAGS,
            "R": (("y", "x"), np.zeros(peaks.shape)),
            "refx": 1,
            "refy": 2,
        },
    )


@pytest.mark.parametrize("gauss_convolve", [False, True])
def test_the_peak_lag_of_every_pixel(gauss_convolve):
    e = event()
    for y in range(NY):
        for x in range(NX):
            tau = get_maximum_time(e, x, y, gauss_convolve=gauss_convolve)
            assert tau == pytest.approx(PEAKS[y, x], abs=1e-12)


def test_the_pixel_defaults_to_the_reference():
    assert get_maximum_time(event()) == pytest.approx(PEAKS[2, 1], abs=1e-12)


@pytest.mark.parametrize("pixel", [(NX, 0), (0, NY), (-1, 0), (0, -1)])
def test_a_pixel_off_the_array_has_no_peak(pixel):
    assert get_maximum_time(event(), *pixel) is None


def test_a_peak_at_the_window_edge_warns():
    e = event(peaks=np.full((NY, NX), 20e-6))  # past the last lag
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tau = get_maximum_time(e, 0, 0)
    assert tau == LAGS[-1]
    messages = [str(w.message) for w in caught]
    assert any("window edge" in m for m in messages)
    assert any("maximum in the boundary" in m for m in messages)


def test_gaussian_convolve_keeps_what_the_whole_kernel_covers():
    times = np.arange(50.0)
    smoothed_times, smoothed = gaussian_convolve(np.ones(50), times, s=3)
    # 6 s = 18 samples, made odd: 19, so 9 lost at either end.
    np.testing.assert_array_equal(smoothed_times, times[9:-9])
    np.testing.assert_allclose(smoothed, 1.0, rtol=1e-15)  # the kernel sums to 1
    t, _ = gaussian_convolve(np.ones(50), times, s=1.0, kernel_size=5)
    np.testing.assert_array_equal(t, times[2:-2])


def test_find_maximum_interpolate_between_samples():
    x = np.arange(10.0)
    y = -((x - 4.3) ** 2)
    x_max, y_max = find_maximum_interpolate(x, y)
    assert x_max == pytest.approx(4.3, abs=1e-12)
    assert y_max == pytest.approx(0.0, abs=1e-12)
