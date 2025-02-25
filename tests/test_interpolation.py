# from read_apd import *
import xarray as xr
import numpy as np
import phantom as ph

# Reconstruct the xarray DataArray with interpolated values
data = []
x = np.arange(0, np.pi, 0.1)
y = np.arange(0, np.pi, 0.1)
time = np.arange(0, 10, 1)

M = np.sin(np.add.outer(x, y))
for t in time:
    data.append(t * M)

array = xr.DataArray(
    np.stack(data, axis=0),
    dims=("time", "y", "x"),
    coords={"time": time, "y": y, "x": x},
)

ds = xr.Dataset({"frames": array})
ds["frames"].values[:, 5, 5] = np.nan
ds = ph.interpolate_nans_3d(ds)

error = ds["frames"].values[:, 5, 5] - time * np.sin(1)


def test_interpolation():
    assert np.all(error < 0.1), "Error too large"
