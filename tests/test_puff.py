"""imaging_methods.puff on synthetic array means whose puff is known.

Moved from fusion_scripts `tests/test_puff.py` at 12b8543; the test of the drivers that
read files (``shot_window``, ``table_row`` and the command line) stays there.
"""

import numpy as np
import pytest
import xarray as xr

from imaging_methods import puff

DT = 1e-5  # 100 kHz: a 1 ms running mean is 100 samples
DARK, ON = -1.1, -0.8  # volts: C-Mod reads more light higher
T_END = 0.5


def signal(
    rise_at=0.1,
    rise_time=0.02,
    off_at=None,
    dips=(),
    noise=2e-4,
    seed=0,
    dark=DARK,
    on=ON,
    t_end=T_END,
):
    """``(time, mean)``: dark until ``rise_at``, a linear ramp over ``rise_time``, the plateau, the light gone at
    ``off_at``, and each ``(a, b)`` of ``dips`` back at the dark level. The ramp is at half height at
    ``rise_at + rise_time / 2``."""
    t = np.arange(0, t_end, DT)
    x = (
        np.clip((t - rise_at) / rise_time, 0, 1)
        if rise_at is not None
        else np.ones_like(t)
    )
    if off_at is not None:
        x = np.where(t >= off_at, 0.0, x)
    for a, b in dips:
        x = np.where((t >= a) & (t < b), 0.0, x)
    mean = dark + (on - dark) * x
    return t, mean + noise * np.random.default_rng(seed).normal(size=t.size)


HALF = 0.1 + 0.02 / 2  # where the default ramp crosses the threshold


def test_running_mean_matches_a_convolution():
    x = np.random.default_rng(1).normal(size=500) + 100
    np.testing.assert_allclose(
        puff.running_mean(x, 37), np.convolve(x, np.ones(37) / 37, "valid"), rtol=1e-12
    )


def test_the_puff_starts_where_the_signal_crosses_halfway():
    t, x = signal()
    w = puff.find_puff(t, x, (0.0, 0.49))
    assert w.found
    assert w.start == pytest.approx(HALF, abs=1.5e-3)
    assert w.analysis_window[0] == pytest.approx(HALF, abs=1.5e-3)
    assert w.analysis_window[1] == 0.49  # the light stays on to the end of the window
    assert w.baseline == pytest.approx(DARK, abs=1e-3)
    assert w.level == pytest.approx(ON, abs=1e-3)
    assert w.threshold == pytest.approx((DARK + ON) / 2, abs=1e-3)
    assert w.dips == ()
    assert w.cut == pytest.approx(HALF / 0.49, abs=0.01)
    assert "puff rises at" in w.note and "dark" in w.note
    assert (
        w.level_quantile == 0.5 and not w.fallback and "percentile" not in w.note
    )  # the median, as the rule says


def test_a_window_that_starts_after_the_puff_is_not_cut():
    t, x = signal()
    w = puff.find_puff(t, x, (0.2, 0.49))
    assert w.found
    assert w.analysis_window == (0.2, 0.49)
    assert w.start == pytest.approx(HALF, abs=1.5e-3)  # found, though before the window
    assert "already on at the window's start" in w.note


def test_a_cut_shorter_than_the_smoothing_window_is_not_made():
    t, x = signal(rise_at=0.1004, rise_time=1e-3)
    w = puff.find_puff(t, x, (0.1, 0.49))
    assert w.found and w.analysis_window == (0.1, 0.49)


def test_the_puff_ends_where_the_light_goes():
    t, x = signal(off_at=0.35)
    w = puff.find_puff(t, x, (0.05, 0.4))
    assert w.found
    assert w.end == pytest.approx(0.35, abs=1.5e-3)
    assert w.analysis_window[1] == pytest.approx(0.35, abs=1.5e-3)
    assert "light gone" in w.note


def test_a_dip_inside_the_window_is_reported_and_not_cut():
    t, x = signal(dips=[(0.30, 0.315)])
    w = puff.find_puff(t, x, (0.0, 0.49))
    assert w.found and w.analysis_window[1] == 0.49
    assert len(w.dips) == 1
    assert w.dips[0][0] == pytest.approx(0.30, abs=2e-3) and w.dips[0][
        1
    ] == pytest.approx(0.315, abs=2e-3)
    assert "dip below threshold at 0.3" in w.note


def test_a_dip_shorter_than_a_millisecond_is_not_reported():
    t, x = signal(dips=[(0.30, 0.3004)])
    assert puff.find_puff(t, x, (0.0, 0.49)).dips == ()


def test_the_polarity_is_not_assumed():
    t, x = signal()
    up = puff.find_puff(t, x, (0.0, 0.49))
    down = puff.find_puff(t, -x, (0.0, 0.49))
    assert down.found
    assert down.analysis_window == pytest.approx(up.analysis_window)
    assert down.baseline == pytest.approx(-up.baseline)


def test_a_blip_shorter_than_five_milliseconds_is_not_the_puff():
    t, x = signal()
    x = x + np.where(
        (t > 0.03) & (t < 0.033), 0.4, 0.0
    )  # 3 ms above the threshold, long before the rise
    w = puff.find_puff(t, x, (0.0, 0.49))
    assert w.found and w.start == pytest.approx(HALF, abs=1.5e-3)


def test_a_record_that_starts_mid_rise_has_no_dark_level():
    t, x = signal(
        rise_at=-0.003, rise_time=0.015
    )  # a fifth of the way up at the first sample, still rising
    w = puff.find_puff(t, x, (0.0, 0.49))
    assert not w.found
    assert w.analysis_window == (0.0, 0.49) and w.cut == 0
    assert "already on at the record's start" in w.note
    assert np.isnan(w.start) and np.isnan(w.end)


def test_a_record_that_starts_with_the_light_on_is_kept():
    t, x = signal(rise_at=None)
    w = puff.find_puff(t, x, (0.0, 0.49))
    assert not w.found and w.analysis_window == (0.0, 0.49)
    assert "no clear rise" in w.note and "starts with the light on" in w.note


def test_a_mostly_dark_window_takes_the_90th_percentile_as_the_level():
    # On 0.11-0.20 s of a 0.49 s window: the median over the window is the dark level, so the median has no rise.
    t, x = signal(off_at=0.2)
    w = puff.find_puff(t, x, (0.0, 0.49))
    assert w.found and w.fallback and w.level_quantile == 0.9
    assert w.level == pytest.approx(
        ON, abs=2e-3
    )  # the plateau, where the window's top tenth is
    assert w.threshold == pytest.approx((DARK + ON) / 2, abs=2e-3)
    assert w.analysis_window[0] == pytest.approx(HALF, abs=1.5e-3)
    assert w.analysis_window[1] == pytest.approx(0.2, abs=1.5e-3)
    assert (
        "its median over the discharge window" not in w.note
    )  # it names the level it took, and the median's
    assert (
        "the median over the discharge window (-1.1" in w.note
        and "90th percentile (-0.8" in w.note
    )
    assert "most of the window is dark" in w.note and "light gone at 0.2" in w.note
    assert (
        "its 90th percentile where the median is not clearly above the dark level"
        in w.rule
    )  # the file says so
    assert w.note[1:] in w.rule and w.rule.endswith(w.note[1:])


def test_without_the_fallback_a_mostly_dark_window_is_kept():
    # The rule with the median alone: no rise, so the discharge window stays.
    t, x = signal(off_at=0.2)
    w = puff.find_puff(t, x, (0.0, 0.49), fallback_quantile=None)
    assert (
        not w.found
        and w.analysis_window == (0.0, 0.49)
        and w.level_quantile == 0.5
        and not w.fallback
    )
    assert "no clear rise" in w.note and "most of the window is dark" in w.note


def test_the_fallback_changes_nothing_where_the_median_passes():
    for kwargs, window in (
        ({}, (0.0, 0.49)),
        ({"off_at": 0.35, "dips": [(0.2, 0.22)]}, (0.05, 0.4)),
        ({}, (0.2, 0.49)),
    ):
        t, x = signal(**kwargs)
        with_fallback, median_only = puff.find_puff(t, x, window), puff.find_puff(
            t, x, window, fallback_quantile=None
        )
        assert with_fallback.found and not with_fallback.fallback
        for name in (
            "analysis_window",
            "start",
            "end",
            "baseline",
            "level",
            "threshold",
            "dips",
            "note",
            "rule",
        ):
            assert getattr(with_fallback, name) == getattr(median_only, name), name


def test_the_fallback_takes_the_other_tail_where_light_reads_lower():
    t, x = signal(off_at=0.2)
    up, down = puff.find_puff(t, x, (0.0, 0.49)), puff.find_puff(t, -x, (0.0, 0.49))
    assert down.found and down.fallback and down.level_quantile == pytest.approx(0.1)
    assert down.analysis_window == pytest.approx(up.analysis_window)
    assert (
        down.level == pytest.approx(-up.level)
        and "10th percentile" in down.note
        and "below" in down.note
    )


def lit_records(seed=4):
    """Two records that start with the light on, whose median is the level they start at, so the median has no
    rise, and with a large excursion in the other tail of the window, to which the fallback would take the level:
    a bump up, and a drop at the end (a decay, read the other way up). Their start varies by tenths of the rise,
    as the plasma's own fluctuations do, where a dark level is flat to under 4%."""
    t = np.arange(0, T_END, DT)
    noise = 0.5 * np.random.default_rng(seed).normal(size=t.size)
    bump = 0.4 + 0.6 * ((t > 0.30) & (t < 0.38)) + noise
    drop = 0.4 - 0.4 * (t > 0.34) + noise
    return {"bump": (t, bump), "drop": (t, drop)}


@pytest.mark.parametrize("name", ["bump", "drop"])
def test_a_record_that_starts_lit_does_not_take_the_fallback(name):
    t, x = lit_records()[name]
    w = puff.find_puff(t, x, (0.0, 0.49))
    # The median fails the clear-rise test, and the fallback level would pass it ...
    assert w.rise <= 0 or w.rise < puff.CLEAR_RISE * w.excursion
    # ... but the start of the record is not flat against the rise to that level, so the median's verdict stands
    assert not w.found and not w.fallback and w.level_quantile == 0.5
    assert w.analysis_window == (0.0, 0.49) and w.note.startswith(
        "no clear rise: the median over the discharge window"
    )
    # Without the flat-start test the fallback would take it, and give a window with no physical sense
    unguarded = puff.find_puff(t, x, (0.0, 0.49), flat_start=np.inf)
    assert unguarded.found and unguarded.fallback
    if name == "drop":
        assert (
            unguarded.analysis_window[0] > 0.3
        )  # the record's decaying tail, with the light going down


def test_a_window_before_the_puff_is_kept():
    t, x = signal(rise_at=0.3)
    w = puff.find_puff(t, x, (0.05, 0.2))
    assert not w.found and w.analysis_window == (0.05, 0.2)


def test_a_puff_that_barely_overlaps_the_window_is_kept():
    t, x = signal(
        off_at=0.30
    )  # on from 0.11 to 0.30; the window starts at 0.29 and the median is dark
    w = puff.find_puff(t, x, (0.29, 0.49))
    assert not w.found and w.analysis_window == (0.29, 0.49)


def test_the_rule_in_words_says_what_was_done_or_why_not():
    t, x = signal()
    found = puff.find_puff(t, x, (0.0, 0.49))
    assert (
        found.rule == f"{puff.RULE} {found.note[0].upper()}{found.note[1:]}"
    )  # a sentence after the rule's
    kept = puff.find_puff(t, x * 0 + x[0], (0.0, 0.49))
    assert kept.rule.startswith("Discharge window kept: ") and kept.note in kept.rule


def test_bad_input_is_refused():
    t, x = signal()
    with pytest.raises(ValueError, match="outside the record"):
        puff.find_puff(t, x, (1.0, 2.0))
    with pytest.raises(ValueError, match="empty"):
        puff.find_puff(t, x, (0.3, 0.3))
    with pytest.raises(ValueError, match="non-finite"):
        puff.find_puff(t, np.where(t > 0.4, np.nan, x), (0.0, 0.49))
    with pytest.raises(ValueError, match="same length"):
        puff.find_puff(t, x[:-1], (0.0, 0.49))


def test_the_signal_kept_for_drawing_is_the_smoothed_array_mean():
    t, x = signal()
    w = puff.find_puff(t, x, (0.0, 0.49))
    assert w.time.size == w.signal.size <= puff.MAX_POINTS
    assert w.record_window == (t[0], t[-1])
    assert np.interp(0.05, w.time, w.signal) == pytest.approx(DARK, abs=2e-3)
    assert np.interp(0.4, w.time, w.signal) == pytest.approx(ON, abs=2e-3)


def frames_from(mean, gains, seed=2):
    """A (y, x, time) record whose pixels are the mean times a gain, plus noise of their own."""
    rng = np.random.default_rng(seed)
    ny, nx = gains.shape
    dark = DARK
    data = (
        dark
        + gains[..., None] * (mean - dark)
        + 1e-3 * rng.normal(size=(ny, nx, mean.size))
    )
    return data


def test_array_mean_skips_pixels_that_are_not_finite_over_the_record():
    t, x = signal()
    data = frames_from(x, np.ones((3, 4))).astype(np.float32)
    data[1, 2, 100] = np.nan  # one bad sample: the pixel is out of the mean
    data[0, 0] = np.nan
    frames = xr.DataArray(data, dims=("y", "x", "time"), coords={"time": t})
    keep = np.ones((3, 4), bool)
    keep[1, 2] = keep[0, 0] = False
    np.testing.assert_allclose(
        puff.array_mean(frames), data[keep].astype(np.float64).mean(0), rtol=1e-12
    )
    with pytest.raises(ValueError, match="no pixel"):
        puff.array_mean(frames * np.nan)


def test_puff_window_reads_a_record():
    t, x = signal()
    gains = np.linspace(0.5, 1.5, 12).reshape(3, 4)
    data = frames_from(x, gains)
    data[2, 3] = np.nan  # a dead pixel, as the older raw files have
    ds = xr.Dataset({"frames": (("y", "x", "time"), data)}, coords={"time": t})
    w = puff.puff_window(ds, (0.0, 0.49))
    assert w.found and w.start == pytest.approx(HALF, abs=1.5e-3)
    assert (
        puff.puff_window(ds["frames"], (0.0, 0.49)).analysis_window == w.analysis_window
    )
    same = puff.find_puff(t, puff.array_mean(ds["frames"]), (0.0, 0.49))
    assert same.analysis_window == w.analysis_window and same.threshold == w.threshold


@pytest.mark.parametrize(
    "kwargs, window, found",
    [
        ({}, (0.0, 0.49), True),
        ({"off_at": 0.35, "dips": [(0.2, 0.22)]}, (0.05, 0.4), True),
        ({"rise_at": None}, (0.0, 0.49), False),
        ({"off_at": 0.2}, (0.0, 0.49), True),  # the fallback's level
    ],
)
def test_a_result_survives_a_dataset_and_netcdf(tmp_path, kwargs, window, found):
    t, x = signal(**kwargs)
    w = puff.find_puff(t, x, window)
    assert w.found is found
    assert bool(w.dips) is bool(kwargs.get("dips"))
    path = tmp_path / "puff.nc"
    w.to_dataset().to_netcdf(path)
    with xr.open_dataset(path) as ds:
        back = puff.PuffWindow.from_dataset(ds.load())
    assert back.found is w.found
    assert (
        back.analysis_window == w.analysis_window
        and back.discharge_window == w.discharge_window
    )
    assert back.note == w.note and back.rule == w.rule and back.cut == w.cut
    assert (
        back.level == w.level
        and back.level_quantile == w.level_quantile
        and back.fallback is w.fallback
    )
    assert back.dips == w.dips
    np.testing.assert_array_equal(back.signal, w.signal)
    np.testing.assert_array_equal(back.time, w.time)
    assert (back.start == w.start) or (np.isnan(back.start) and np.isnan(w.start))
