import numpy as np
import xarray as xr
import pytest
from imaging_methods import (
    rotated_blob,
    ellipse_parameters,
    fit_ellipse,
    fit_ellipse_to_event,
    estimate_fwhm_sizes,
)


def make_grid(nx=51, ny=41):
    # Create 2D coordinate grids R (x) and Z (y) with dims ('x', 'y')
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing="ij")  # shapes (nx, ny)
    return X.astype(float), Y.astype(float)


def make_dataarray_from_blob(lx, ly, theta, refx, refy, nx=51, ny=41):
    R, Z = make_grid(nx, ny)
    rx, ry = R[refx, refy], Z[refx, refy]
    data_vals = rotated_blob([lx, ly, theta], rx, ry, R, Z)
    # Build an xarray DataArray with dims ('x', 'y', 'time') and coords R, Z (2D)
    da = xr.DataArray(
        np.stack([data_vals], axis=-1),  # add time dim of length 1
        dims=("x", "y", "time"),
        coords={
            "R": (("x", "y"), R),
            "Z": (("x", "y"), Z),
            "time": [0],
        },
        name="cond_av",
    )
    return da


def make_dataset_for_fwhm(cond_av_da, refx, refy):
    # Build a dataset structure expected by estimate_fwhm_sizes
    ds = xr.Dataset(
        {
            "cond_av": cond_av_da,
            "R": cond_av_da.coords["R"],
            "Z": cond_av_da.coords["Z"],
            "refx": xr.DataArray(refx),
            "refy": xr.DataArray(refy),
        }
    )
    return ds


def test_rotated_blob_center_is_one():
    nx, ny = 16, 16
    refx, refy = nx // 2, ny // 2
    lx, ly, theta = 3.0, 5.0, 0.0
    R, Z = make_grid(nx, ny)
    rx, ry = R[refx, refy], Z[refx, refy]
    blob = rotated_blob([lx, ly, theta], rx, ry, R, Z)
    assert blob.shape == (nx, ny)
    assert np.isclose(blob[refx, refy], 1.0), "Blob value at the center should be 1"


def test_rotated_blob_rotation_swaps_axes_at_pi_over_2():
    nx, ny = 16, 16
    refx, refy = nx // 2, ny // 2
    R, Z = make_grid(nx, ny)
    rx, ry = R[refx, refy], Z[refx, refy]
    lx, ly = 2.0, 5.0
    # Unrotated: extents along x and y differ
    blob0 = rotated_blob([lx, ly, 0.0], rx, ry, R, Z)
    # Rotated by pi/2 should swap the roles of lx and ly along axes
    blob90 = rotated_blob([lx, ly, np.pi / 2], rx, ry, R, Z)
    # Compare slices through center: x-axis vs y-axis swapped
    center_x_slice_blob0 = blob0[:, refy]  # vary x, fix y
    center_y_slice_blob0 = blob0[refx, :]  # vary y, fix x
    center_x_slice_blob90 = blob90[:, refy]
    center_y_slice_blob90 = blob90[refx, :]
    # After 90-degree rotation, profiles swap
    assert np.allclose(center_x_slice_blob0, center_y_slice_blob90, atol=1e-6)
    assert np.allclose(center_y_slice_blob0, center_x_slice_blob90, atol=1e-6)


def test_ellipse_parameters_theta_zero_parametrization():
    lx, ly, theta = 3.0, 5.0, 0.0
    rx, ry = 10.0, 20.0
    alphas = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    xvals, yvals = ellipse_parameters([lx, ly, theta], rx, ry, alphas)
    # For theta = 0, x = rx + lx*cos(alpha), y = ry + ly*sin(alpha)
    expected_x = rx + lx * np.cos(alphas)
    expected_y = ry + ly * np.sin(alphas)
    assert np.allclose(xvals, expected_x, atol=1e-12)
    assert np.allclose(yvals, expected_y, atol=1e-12)


@pytest.mark.parametrize(
    "lx_true, ly_true, theta_true",
    [
        (2.5, 4.0, 0.2),  # lx < ly
        (3.0, 5.0, 0.0),  # unrotated
        (1.5, 3.5, 0.8),  # larger theta
        (2.0, 2.0, 0),  # circular
    ],
)
def test_fit_ellipse_recovers_parameters_diff_evo(lx_true, ly_true, theta_true):
    # Create synthetic data with known parameters
    nx, ny = 16, 16
    refx, refy = nx // 2, ny // 2
    e = make_dataarray_from_blob(lx_true, ly_true, theta_true, refx, refy, nx=nx, ny=ny)
    rx, ry = e.R.sel(x=refx, y=refy).item(), e.Z.sel(x=refx, y=refy).item()
    lx, ly, theta = fit_ellipse(
        e.sel(time=0),
        rx,
        ry,
        size_penalty_factor=0.0,
        aspect_ratio_penalty_factor=0.0,
        theta_penalty_factor=0.1,
    )

    assert 0.0 <= theta <= np.pi
    assert lx <= ly, "By convention lx <= ly"
    assert np.isclose(lx, lx_true, rtol=0.15, atol=0.3)
    assert np.isclose(ly, ly_true, rtol=0.15, atol=0.3)
    assert np.isclose(theta % np.pi, theta_true, rtol=0.3, atol=0.3)


def test_fit_ellipse_convention_swap_when_lx_gt_ly():
    # If lx_true > ly_true, the function should return swapped axes and theta + pi/2
    nx, ny = 16, 16
    refx, refy = nx // 2, ny // 2
    lx_true, ly_true, theta_true = 6.0, 2.0, 0.0
    e = make_dataarray_from_blob(lx_true, ly_true, theta_true, refx, refy, nx=nx, ny=ny)
    lx, ly, theta = fit_ellipse_to_event(
        e,
        refx=refx,
        refy=refy,
        size_penalty_factor=0.0,
        aspect_ratio_penalty_factor=0.0,
        theta_penalty_factor=0.0,
    )
    assert lx <= ly
    # When original lx > ly and theta = 0, the convention enforces theta ~ pi/2
    assert np.isclose(theta, np.pi / 2, atol=0.3)
    # Axes are swapped
    assert np.isclose(lx, ly_true, rtol=0.15, atol=0.3)
    assert np.isclose(ly, lx_true, rtol=0.15, atol=0.3)


def test_estimate_fwhm_sizes_gaussian_unrotated():
    # For an unrotated Gaussian: f(y) = exp(-( y^2 / ly^2 ))
    # FWHM f satisfies exp(-(f/2)^2 / ly^2)) = 1/2  => d = 2 * ly * sqrt(ln(2)) = 0.83 * ly
    nx, ny = 16, 16
    refx, refy = nx // 2, ny // 2
    lx_true, ly_true, theta_true = 4.0, 6.0, 0.0
    cond_av_da = make_dataarray_from_blob(
        lx_true, ly_true, theta_true, refx, refy, nx=nx, ny=ny
    )
    ds = make_dataset_for_fwhm(cond_av_da, refx, refy)
    rp_fwhm, rn_fwhm, zp_fwhm, zn_fwhm = estimate_fwhm_sizes(ds)
    expected_r = lx_true * 0.83
    expected_z = ly_true * 0.83
    # Check signs and approximate magnitudes (discretization + interpolation may introduce small error)
    assert rp_fwhm >= 0 >= rn_fwhm
    assert zp_fwhm >= 0 >= zn_fwhm
    assert np.isclose(rp_fwhm, expected_r, atol=0.5)
    assert np.isclose(-rn_fwhm, expected_r, atol=0.5)
    assert np.isclose(zp_fwhm, expected_z, atol=0.5)
    assert np.isclose(-zn_fwhm, expected_z, atol=0.5)
