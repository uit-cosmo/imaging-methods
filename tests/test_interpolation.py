# from read_apd import *
import xarray as xr
import numpy as np
import phantom as ph

# Reconstruct the xarray DataArray with interpolated values
x = np.arange(0, np.pi, 0.1)
y = np.arange(0, np.pi, 0.1)
time = np.arange(0, np.pi, 0.1)

M = np.sin(np.add.outer(x, y))
data = [np.sin(t) * M for t in time]

array = xr.DataArray(
    np.stack(data, axis=0),
    dims=("time", "y", "x"),
    coords={"time": time, "y": y, "x": x},
)

ds = xr.Dataset({"frames": array})
ds["frames"].values[:, 5, 5] = np.nan
ds["frames"].values[:, 5, 4] = np.nan


def test_interpolation():
    ph.interpolate_nans_3d(ds)
    error = np.abs(ds["frames"].values[:, 5, 5] - np.sin(time) * np.sin(1))

    assert np.all(error < 0.1), "Error too large"
