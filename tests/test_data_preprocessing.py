import xarray as xr
import numpy as np
import phantom as ph
import phantom.data_preprocessing
import unittest

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


def test_interpolationn():
    ds_new = phantom.data_preprocessing.interpolate_nans_3d(ds)
    error = np.abs(ds_new["frames"].values[5, 5, :] - np.sin(time) * np.sin(1))

    assert np.all(error < 0.1), "Error too large"
