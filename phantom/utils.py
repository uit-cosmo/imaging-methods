import fppanalysis as fppa
import velocity_estimation as ve
import xarray as xr
import numpy as np


def run_norm_ds(ds, radius):
    """Returns running normalized dataset of a given dataset using run_norm from
    fppanalysis function by applying xarray apply_ufunc.
    Input:
        - ds: xarray Dataset
        - radius: radius of the window used in run_norm. Window size is 2*radius+1. ... int
    'run_norm' returns a tuple of time base and the signal. Therefore, apply_ufunc will
    return a tuple of two DataArray (corresponding to time base and the signal).
    To return a format like the original dataset, we create a new dataset of normalized frames and
    corresponding time computed from apply_ufunc.
    Description of apply_ufunc arguments.
        - first the function
        - then arguments in the order expected by 'run_norm'
        - input_core_dimensions: list of lists, where the number of inner sequences must match
        the number of input arrays to the function 'run_norm'. Each inner sequence specifies along which
        dimension to align the corresponding input argument. That means, here we want to normalize
        frames along time, hence 'time'.
        - output_core_dimensions: list of lists, where the number of inner sequences must match
        the number of output arrays to the function 'run_norm'.
        - exclude_dims: dimensions allowed to change size. This must be set for some reason.
        - vectorize must be set to True in order to for run_norm to be applied on all pixels.
    """
    import xarray as xr

    normalization = xr.apply_ufunc(
        fppa.run_norm,
        ds["frames"],
        radius,
        ds["time"],
        input_core_dims=[["time"], [], ["time"]],
        output_core_dims=[["time"], ["time"]],
        exclude_dims=set(("time",)),
        vectorize=True,
    )

    ds_normalized = xr.Dataset(
        data_vars=dict(
            frames=(["y", "x", "time"], normalization[0].data),
        ),
        coords=dict(
            time=normalization[1].data[0, 0, :],
        ),
    )

    return ds_normalized


def interpolate_nans_3d(ds, time_dim="time"):
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
    from scipy.interpolate import griddata

    def interpolate_2d(array, x, y):
        """Interpolate a 2D array with NaNs using griddata."""
        valid_mask = ~np.isnan(array)
        if np.sum(valid_mask) < 4:
            return array  # Not enough points to interpolate reliably

        valid_points = np.array((x[valid_mask], y[valid_mask])).T
        valid_values = array[valid_mask]

        nan_points = np.array((x[~valid_mask], y[~valid_mask])).T

        interpolated_values = griddata(
            valid_points, valid_values, nan_points, method="linear"
        )
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
        coords={time_dim: ds[time_dim], "y": ds["y"], "x": ds["x"]},
    )

    return xr.Dataset({"frames": interpolated_array})


class PhantomDataInterface(ve.ImagingDataInterface):
    """Implementation of ImagingDataInterface for xarray datasets given by the
    code at https://github.com/sajidah-ahmed/cmod_functions."""

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def get_shape(self):
        return self.ds.dims["x"], self.ds.dims["y"]

    def get_signal(self, x: int, y: int):
        return self.ds.isel(x=x, y=y)["frames"].values

    def get_dt(self) -> float:
        times = self.ds["time"]
        return float(times[1].values - times[0].values)

    def get_position(self, x: int, y: int):
        return x, y

    def is_pixel_dead(self, x: int, y: int):
        signal = self.get_signal(x, y)
        return len(signal) == 0 or np.isnan(signal[0])


def get_t_start_end(ds):
    times = ds.time.values
    t_start = times[0]
    t_end = times[len(times) - 1]
    return t_start, t_end
