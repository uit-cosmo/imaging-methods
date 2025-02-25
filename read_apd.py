import xarray as xr
from show_data import *
from utils import *
import velocity_estimation as ve
import cosmoplots as cp
import matplotlib.pyplot as plt




import xarray as xr
import numpy as np
from scipy.interpolate import griddata


def interpolate_nans_3d(ds, spatial_dims=('x', 'y'), time_dim='time'):
    """
    Replace NaN values in a 3D xarray dataset with linear interpolation
    based on neighboring values.

    Parameters:
        ds (xarray.Dataset or xarray.DataArray): Input dataset or data array.
        spatial_dims (tuple): Names of the two spatial dimensions (default: ('x', 'y')).
        time_dim (str): Name of the time dimension (default: 'time').

    Returns:
        xarray.DataArray: Dataset with NaNs replaced by interpolated values.
    """

    def interpolate_2d(array, x, y):
        """Interpolate a 2D array with NaNs using griddata."""
        valid_mask = ~np.isnan(array)
        if np.sum(valid_mask) < 4:
            return array  # Not enough points to interpolate reliably

        valid_points = np.array((x[valid_mask], y[valid_mask])).T
        valid_values = array[valid_mask]

        nan_points = np.array((x[~valid_mask], y[~valid_mask])).T

        interpolated_values = griddata(valid_points, valid_values, nan_points, method='linear')
        array[~valid_mask] = interpolated_values
        return array

    x, y = np.meshgrid(ds["x"], ds["y"], indexing="xy")

    # Iterate over the time dimension and interpolate each 2D slice
    interpolated_data = []
    for t in ds[time_dim]:
        slice_data = ds["frames"].sel({time_dim: t}).values
        interpolated_slice = interpolate_2d(slice_data, x, y)
        interpolated_data.append(interpolated_slice)

    # Reconstruct the xarray DataArray with interpolated values
    interpolated_array = xr.DataArray(
        np.stack(interpolated_data, axis=0),
        dims=(time_dim, "y", "x"),
        coords={time_dim: ds[time_dim], "y": ds["y"], "x": ds["x"]}
    )

    return xr.Dataset({"frames": interpolated_array})


shot = 1140613026

ds = xr.open_dataset("data/apd_{}.nc".format(shot))
ds = run_norm_ds(ds, 1000)

ds = ds.fillna(0)

t_start, t_end = get_t_start_end(ds)
print("Data with times from {} to {}".format(t_start, t_end))

t_start = (t_start + t_end) / 2
t_end = t_start + 0.001
ds = ds.sel(time=slice(t_start, t_end))
ds = interpolate_nans_3d(ds)

dt = get_dt(ds)

show_movie(ds, variable="frames", gif_name="movie_apd_{}.gif".format(shot), fps=60)
