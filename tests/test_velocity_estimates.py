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
