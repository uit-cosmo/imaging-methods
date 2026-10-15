"""imaging_methods.dead_pixels on a synthetic record whose dead pixels are known.

Moved from fusion_scripts `tests/test_dead_pixels.py` at 12b8543; the tests of the
drivers that read files (``estimate_shot`` and the rest) stay there. The last test here
is ``estimate_shot``'s first, on a record in hand: the puff window, then the estimate
over it.
"""

import numpy as np
import pytest
import xarray as xr
from scipy import signal

from imaging_methods import dead_pixels as dp
from imaging_methods import puff

DT = 5e-7
N = 200_000


def _ar1(rng, tau=20e-6):
    """Red noise of unit variance with correlation time ``tau``: the plasma's
    stand-in."""
    phi = np.exp(-DT / tau)
    return signal.lfilter([1], [1, -phi], rng.normal(size=N) * np.sqrt(1 - phi**2))


@pytest.fixture(scope="module")
def record():
    """4 x 5 pixels sharing one red signal, with one pixel of each kind the method must
    tell apart:

    (1, 1) dim live: a fifth of the shared signal under more noise, so its spectrum
        alone is not red enough;
    (3, 4) dead, white noise only;
    (0, 4) dead with crosstalk: 3% of its neighbour (0, 3) under noise, as a dead
        channel next to a bright one;
    (2, 0) no data.
    """
    rng = np.random.default_rng(0)
    shared = _ar1(rng)
    frames = np.empty((4, 5, N))
    for y in range(4):
        for x in range(5):
            frames[y, x] = 0.9 * shared + 0.4 * _ar1(rng) + 0.02 * rng.normal(size=N)
    frames[1, 1] = 0.2 * shared + 0.12 * rng.normal(size=N)
    frames[3, 4] = 0.01 * rng.normal(size=N)
    frames[0, 4] = 0.03 * frames[0, 3] + 0.015 * rng.normal(size=N)
    frames[2, 0] = np.nan
    return xr.Dataset(
        {"frames": (("y", "x", "time"), frames)}, coords={"time": np.arange(N) * DT}
    )


@pytest.fixture(scope="module")
def result(record):
    return dp.estimate(record, keep_psd=True)


def test_bright_pixels_are_live_by_their_spectrum(result):
    ev = result.evidence.values
    bright = np.ones(ev.shape, bool)
    for y, x in ((1, 1), (3, 4), (0, 4), (2, 0)):
        bright[y, x] = False
    assert (ev[bright] == dp.RED_SPECTRUM).all()


def test_a_dim_pixel_is_live_through_its_neighbours(result):
    assert (
        float(result.red_ratio[1, 1]) < dp.RED
    )  # the case the neighbour step exists for
    assert int(result.evidence[1, 1]) == dp.FOLLOWS_NEIGHBOUR
    assert not bool(result.dead[1, 1])


def test_white_noise_and_crosstalk_are_dead(result):
    for y, x in ((3, 4), (0, 4)):
        assert bool(result.dead[y, x]), (y, x)
        assert int(result.evidence[y, x]) == dp.NO_EVIDENCE
    # Crosstalk follows its neighbour perfectly, at the coupling: the gain, not the
    # correlation, rejects it.
    assert float(result.gain[0, 4]) == pytest.approx(0.03, abs=0.01)


def test_a_pixel_without_samples_is_dead(result):
    assert bool(result.dead[2, 0])
    assert int(result.evidence[2, 0]) == dp.NO_DATA


def test_the_psd_and_thresholds_travel_with_the_mask(result):
    assert result.psd.dims == ("y", "x", "frequency")
    assert result.attrs["red_threshold"] == dp.RED
    assert result.attrs["gain_threshold"] == dp.GAIN


def test_consolidate_marks_a_pixel_dead_all_day_from_a_third_of_the_shots():
    live, dead = np.zeros((2, 2), bool), np.ones((2, 2), bool)
    assert not dp.consolidate([dead, live, live, live]).any()  # 1 of 4
    assert dp.consolidate([dead, live, live]).all()  # 1 of 3
    assert dp.consolidate([dead]).all()


# ---- the window it judges -------------------------------------------------------------

GATE_AT, GATE_TIME, T_STOP = (
    0.025,
    0.01,
    N * DT,
)  # s: the puff, its ramp, the record's end
DISCHARGE = (0.0, 0.099)


@pytest.fixture(scope="module")
def raw():
    """A raw shot as the server has them (float32, no NaN): the dark level until
    ``GATE_AT``, then the plasma of the ``record`` fixture, with a discharge window that
    is the whole record."""
    n = int(T_STOP / DT)
    rng = np.random.default_rng(3)
    shared = _ar1(rng)[:n]
    t = np.arange(n) * DT
    gate = np.clip((t - GATE_AT) / GATE_TIME, 0, 1)
    frames = np.empty((4, 5, n), np.float32)
    for y in range(4):
        for x in range(5):
            frames[y, x] = (
                -1.1
                + gate * (3.0 + 0.9 * shared + 0.4 * _ar1(rng)[:n])
                + 0.01 * rng.normal(size=n)
            )
    frames[3, 4] = -1.1 + 0.01 * rng.normal(size=n)  # dead: dark all the time
    return xr.Dataset({"frames": (("y", "x", "time"), frames)}, coords={"time": t})


def test_the_window_cut_to_the_puff_is_judged(raw):
    """What fusion_scripts' ``estimate_shot`` does on a raw file: the puff found on the
    whole record, then the estimate over the analysis window."""
    w = puff.puff_window(raw, DISCHARGE)
    a0, a1 = w.analysis_window
    assert w.found and w.rule.startswith("Gas puff on where")
    assert a0 == pytest.approx(GATE_AT + GATE_TIME / 2, abs=2e-3) and a1 == DISCHARGE[1]
    result = dp.estimate(raw.sel(time=slice(a0, a1)))
    assert bool(result.dead[3, 4]) and int(result.dead.sum()) == 1
