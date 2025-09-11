import xarray as xr
import numpy as np
import imaging_methods.data_preprocessing
from numpy.testing import assert_array_equal

# Create test data
x = np.arange(0, np.pi, 0.1)
y = np.arange(0, np.pi, 0.1)
time = np.arange(0, np.pi, 0.1)

Y, X = np.meshgrid(y, x, indexing="ij")
spatial_pattern = np.sin(Y + X)  # Shape: (ny, nx)
temporal_pattern = np.sin(time)  # Shape: (nt,)
data = (
    spatial_pattern[:, :, np.newaxis] * temporal_pattern[np.newaxis, np.newaxis, :]
)  # Shape: (ny, nx, nt)

ds = xr.Dataset(
    data_vars={"frames": (["y", "x", "time"], data)},
    coords={"y": y, "x": x, "time": time},
)
ds["frames"].values[5, 5, :] = np.nan
ds["frames"].values[5, 4, :] = np.nan
ds["frames"].values[0, 0, :] = np.nan
ds["frames"].values[0, 1, :] = np.nan

ds_interpolated = imaging_methods.data_preprocessing.interpolate_nans_3d(ds)


def test_interpolation():
    error = np.abs(ds_interpolated["frames"].values[5, 5, :] - np.sin(time) * np.sin(1))

    assert np.all(error < 0.1), "Error too large"


def test_extrapolation():
    # For extrapolation, we use nearest neighbour value
    error = np.abs(
        ds_interpolated["frames"].values[0, 1, :]
        - ds_interpolated["frames"].values[1, 1, :]
    )

    assert np.all(error < 1e-10), "Error too large"


def test_no_change_if_no_nan():
    # If no pixel is nan the dataset should not be modified
    random_ds = xr.Dataset(
        data_vars={
            "frames": (["y", "x", "time"], np.random.rand(len(x), len(y), len(time)))
        },
        coords={"y": y, "x": x, "time": time},
    )
    copy = random_ds.copy(deep=True)
    random_ds_interpolated = imaging_methods.data_preprocessing.interpolate_nans_3d(
        random_ds
    )

    assert_array_equal(copy.frames.values, random_ds_interpolated.frames.values)
