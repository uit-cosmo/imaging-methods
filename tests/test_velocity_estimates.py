import warnings

import numpy as np
import pytest
import xarray as xr

from imaging_methods import get_averaged_velocity_from_position


def make_position_da(times, r, z):
    return xr.DataArray(
        np.stack([r, z], axis=-1),
        dims=("time", "coord"),
        coords={"time": times, "coord": ["R", "Z"]},
    )


def make_linear_track(v=2.0, w=-0.5, n=21, dt=0.1):
    times = np.arange(n) * dt
    return make_position_da(times, 1 + v * times, -3 + w * times)


@pytest.mark.parametrize("estimator", ["central_diff", "lsq"])
def test_constant_velocity_is_recovered(estimator):
    v_input, w_input = 2.0, -0.5
    position_da = make_linear_track(v_input, w_input)
    mask = np.ones(position_da.sizes["time"], dtype=bool)

    v, w = get_averaged_velocity_from_position(position_da, mask, estimator=estimator)

    assert v == pytest.approx(v_input)
    assert w == pytest.approx(w_input)


@pytest.mark.parametrize("estimator", ["central_diff", "lsq"])
def test_masked_out_times_do_not_contribute(estimator):
    """The track bends outside the mask; the estimators should not see it."""
    v_input, w_input = 2.0, -0.5
    position_da = make_linear_track(v_input, w_input, n=31)
    mask = position_da.time.values < 1.0
    position_da.values[mask.sum() + 1 :, :] *= -5

    v, w = get_averaged_velocity_from_position(position_da, mask, estimator=estimator)

    assert v == pytest.approx(v_input)
    assert w == pytest.approx(w_input)


@pytest.mark.parametrize("estimator", ["central_diff", "lsq"])
def test_non_finite_position_inside_mask_warns(estimator):
    position_da = make_linear_track()
    position_da.values[10, :] = np.nan
    mask = np.ones(position_da.sizes["time"], dtype=bool)

    with pytest.warns(RuntimeWarning, match="non-finite position"):
        get_averaged_velocity_from_position(position_da, mask, estimator=estimator)


def test_lsq_ignores_non_finite_position():
    """lsq drops the bad point and still recovers the velocity exactly."""
    v_input, w_input = 2.0, -0.5
    position_da = make_linear_track(v_input, w_input)
    position_da.values[10, :] = np.nan
    mask = np.ones(position_da.sizes["time"], dtype=bool)

    with pytest.warns(RuntimeWarning):
        v, w = get_averaged_velocity_from_position(position_da, mask, estimator="lsq")

    assert v == pytest.approx(v_input)
    assert w == pytest.approx(w_input)


def test_lsq_is_steadier_than_central_diff_under_noise():
    """
    The centered-difference average telescopes, so it only sees the two ends of the track. Under white position noise
    its error is sigma / (dt (n-2)), against sigma / (dt sqrt(n(n^2-1)/12)) for the slope, so lsq should be a factor
    sqrt(12/n) ~ 0.54 steadier here. Both are unbiased.
    """
    number_of_samples, number_of_lags = 400, 41
    rng = np.random.default_rng(42)
    v_input, w_input = 2.0, -0.5
    errors = {"central_diff": [], "lsq": []}

    for _ in range(number_of_samples):
        position_da = make_linear_track(v_input, w_input, n=number_of_lags)
        position_da.values += rng.normal(scale=0.05, size=position_da.shape)
        mask = np.ones(position_da.sizes["time"], dtype=bool)
        for estimator in errors:
            v, _ = get_averaged_velocity_from_position(
                position_da, mask, estimator=estimator
            )
            errors[estimator].append(v - v_input)

    expected_ratio = np.sqrt(12 / number_of_lags)
    ratio = np.std(errors["lsq"]) / np.std(errors["central_diff"])
    assert ratio == pytest.approx(expected_ratio, rel=0.15)
    for estimator in errors:
        assert abs(np.mean(errors[estimator])) < 0.01


def test_empty_mask_returns_nan():
    position_da = make_linear_track()
    mask = np.zeros(position_da.sizes["time"], dtype=bool)

    v, w = get_averaged_velocity_from_position(position_da, mask)

    assert np.isnan(v) and np.isnan(w)


def test_unknown_estimator_raises():
    position_da = make_linear_track()
    mask = np.ones(position_da.sizes["time"], dtype=bool)

    with pytest.raises(ValueError, match="Unknown velocity estimator"):
        get_averaged_velocity_from_position(position_da, mask, estimator="linear")


def make_moving_gaussian_ds(n_time=21, shape=(9, 9), v=1.0, start=2.0, width=1.2):
    """A Gaussian blob crossing the array, so late frames run off the edge."""
    times = np.arange(n_time) * 1.0
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    frames = np.empty((ny, nx, n_time))
    for i, t in enumerate(times):
        cx = start + v * t
        frames[..., i] = np.exp(-(((xx - cx) ** 2 + (yy - ny // 2) ** 2) / width**2))
    R, Z = np.meshgrid(np.arange(nx) * 0.1, np.arange(ny) * 0.1)
    return xr.Dataset(
        {"cond_av": (["y", "x", "time"], frames)},
        coords={"R": (["y", "x"], R), "Z": (["y", "x"], Z), "time": times},
    ).assign(refx=nx // 2, refy=ny // 2)


def test_within_boundaries_flags_clipped_contours():
    from imaging_methods import get_contour_evolution

    ds = make_moving_gaussian_ds()
    contour_ds = get_contour_evolution(ds["cond_av"], 0.5)
    within = contour_ds.within_boundaries.values

    assert within.dtype == bool
    assert within[0]  # blob starts well inside the array
    assert not within[-1]  # and has run off the far edge by the end


def test_require_within_boundaries_shortens_the_mask():
    from imaging_methods import get_averaged_velocity, get_default_apd_method_params

    ds = make_moving_gaussian_ds()
    params = get_default_apd_method_params()
    params.contouring.threshold_factor = 0.5
    params.position_filter.window_size = 1
    params.position_filter.mask_distance = 100  # let every lag through
    params.position_filter.mask_signal_factor = 0.0

    off = get_averaged_velocity(ds, "cond_av", params)
    params.position_filter.require_within_boundaries = True
    on = get_averaged_velocity(ds, "cond_av", params)

    # The clipped tail drags the velocity down; excluding it recovers the input.
    assert on[0] > off[0]
    assert on[0] == pytest.approx(0.1, rel=0.05)  # v = 1 px/frame, dr = 0.1


# --- get_positions_and_mask: the tracking step, returned ---


def deck_params(estimator="lsq"):
    from imaging_methods import get_default_apd_method_params

    params = get_default_apd_method_params()
    params.position_filter.window_size = 3
    params.position_filter.mask_signal_factor = 0.6
    params.velocity.estimator = estimator
    return params


@pytest.mark.parametrize("estimator", ["central_diff", "lsq"])
@pytest.mark.parametrize("position_method", ["contouring", "max"])
@pytest.mark.parametrize("variable", ["cond_av", "cross_corr"])
def test_positions_and_mask_are_the_steps_of_get_averaged_velocity(
    fresh, variable, position_method, estimator
):
    """On real 2DCA averages (the decorrelation fixture's): the velocity of the track
    and mask it returns is get_averaged_velocity's, to the bit."""
    from imaging_methods import get_averaged_velocity, get_positions_and_mask

    params = deck_params(estimator)
    for pixel in [(2, 1), (2, 3), (0, 4)]:
        average_ds = fresh[pixel]
        pos, mask, valid = get_positions_and_mask(
            average_ds, variable, params, position_method=position_method
        )
        n = average_ds.sizes["time"] - 2  # a smoothing window of 3 loses one each end
        assert pos.dims == ("time", "coord") and pos.shape == (n, 2)
        np.testing.assert_array_equal(pos.time.values, average_ds.time.values[1:-1])
        assert np.asarray(mask).dtype == bool and np.asarray(mask).shape == (n,)
        assert valid.dtype == bool and valid.shape == (n,)
        np.testing.assert_array_equal(valid, pos.valid.values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = get_averaged_velocity_from_position(pos, mask, estimator=estimator)
            expected = get_averaged_velocity(
                average_ds, variable, params, position_method=position_method
            )
        np.testing.assert_array_equal(v, expected)


def test_positions_flag_the_lags_without_a_raw_position():
    """A contour level above the late frames' peaks: no centroid there, so those lags
    are extrapolated, and flagged as not valid."""
    from imaging_methods import get_positions_and_mask

    ds = make_moving_gaussian_ds(v=0.0, start=4.0)  # still, centred on pixel (4, 4)
    ds["cond_av"] = ds.cond_av * np.exp(-ds.time / 10)  # a decaying average
    params = deck_params()
    params.contouring.threshold_factor = 0.4
    params.position_filter.window_size = 1
    pos, mask, valid = get_positions_and_mask(ds, "cond_av", params)
    decayed = np.exp(-ds.time.values / 10) < 0.4  # from lag 10 on
    np.testing.assert_array_equal(valid, ~decayed)
    assert np.isfinite(pos.values).all()  # extrapolated where there was nothing
    np.testing.assert_allclose(pos.values[~decayed], [[0.4, 0.4]] * 10, atol=1e-12)


def test_positions_within_boundaries_and_unknown_methods():
    from imaging_methods import get_positions_and_mask

    ds = make_moving_gaussian_ds()
    params = deck_params()
    params.contouring.threshold_factor = 0.5
    params.position_filter.window_size = 1
    params.position_filter.mask_distance = 100
    params.position_filter.mask_signal_factor = 0.0
    _, everything, _ = get_positions_and_mask(ds, "cond_av", params)
    params.position_filter.require_within_boundaries = True
    _, inside, _ = get_positions_and_mask(ds, "cond_av", params)
    assert np.asarray(everything).all()
    assert np.asarray(inside).sum() < np.asarray(everything).sum()
    assert not np.asarray(inside)[-1]  # the blob has run off the edge by then
    with pytest.raises(NotImplementedError, match="Unknown position method"):
        get_positions_and_mask(ds, "cond_av", params, position_method="centroid")
