import numpy as np
import xarray as xr


def get_test_data():
    values = np.arange(0, 25)
    values.resize(5, 5)
    ds = xr.DataArray(
        values,
        dims=("x", "y"),
        coords={"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]},
    )
    ds = ds.rolling(x=3, y=3, center=True, min_periods=1).mean()
    return ds


ds = xr.open_dataset("data/phantom_1090813019.nc")

window = 1000
print(ds.dims)

# Compute rolling mean and std ignoring NaNs, requiring all 10 values to be non-NaN
rolling = ds.rolling(time=window, center=True, min_periods=window)
rolling_mean = rolling.mean()
rolling_std = ds.rolling(time=window, center=True, min_periods=window).std()

# Normalize: subtract the rolling mean and divide by the rolling std
da_normalized = (ds - rolling_mean) / rolling_std
da_normalized = da_normalized.dropna(dim="time", how="all")
print(ds.dims)
