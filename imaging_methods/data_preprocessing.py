import fppanalysis as fppa
import xarray as xr
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from imaging_methods import get_t_start_end


def interpolate_nans_3d(ds):
    """
    Replace NaN values in a 3D xarray dataset with linear interpolation along spatial dimensions
    using LinearNDInterpolator.

    Parameters:
        ds (xr.Dataset): Input dataset with 'frames' variable (dims: y, x, time).

    Returns:
        xr.Dataset: Dataset with NaNs interpolated.
    """
    # Input validation
    if "frames" not in ds or ds["frames"].dims != ("y", "x", "time"):
        raise ValueError(f"ds must contain 'frames' with dims ('y', 'x', 'time')")
    if ds["frames"].isnull().all():
        raise ValueError("All frame values are NaN")
    if not ds["frames"].isnull().any():
        return ds.copy()

    frames = ds["frames"].values  # Shape: (ny, nx, nt)
    ny, nx, nt = frames.shape

    # Get the NaN mask (constant across time)
    nan_mask = np.isnan(frames[:, :, 0])
    if nan_mask.all():
        raise ValueError("All pixels are NaN in the first time step")

    # Coordinate setup for all points
    y_indices, x_indices = np.indices((ny, nx))
    points = np.vstack((y_indices.ravel(), x_indices.ravel())).T  # Shape: (ny*nx, 2)
    valid_mask = ~nan_mask
    valid_points = points[valid_mask.ravel()]  # Coordinates of non-NaN points
    nan_points = points[nan_mask.ravel()]  # Coordinates of NaN points

    # Process each time step
    for t in range(nt):
        frame = frames[:, :, t].copy()
        valid_values = frame[valid_mask]  # Values at non-NaN points

        # Set up LinearNDInterpolator with valid points and values
        interpolator = LinearNDInterpolator(
            valid_points, valid_values, fill_value=np.nan
        )
        interpolated_values = interpolator(nan_points)

        # Fallback to nearest interpolation for any remaining NaNs
        if np.any(np.isnan(interpolated_values)):
            nearest_interp = NearestNDInterpolator(valid_points, valid_values)
            interpolated_values[np.isnan(interpolated_values)] = nearest_interp(
                nan_points[np.isnan(interpolated_values)]
            )

        # Replace NaN values in the frame
        frame[nan_mask] = interpolated_values
        frames[:, :, t] = frame

    # Update the dataset with interpolated frames
    ds["frames"] = xr.DataArray(
        frames,
        dims=ds["frames"].dims,
        coords=ds["frames"].coords,
        attrs=ds["frames"].attrs,
    )
    return ds


def run_norm_ds(ds, radius):
    """
    Compute running normalized dataset using fppanalysis.run_norm.

    Parameters:
        ds (xr.Dataset): Input dataset with 'frames' (dims: y, x, time) and 'time' coordinate.
        radius (int): Radius of the normalization window (window size = 2*radius+1 time steps).

    Returns:
        xr.Dataset: Normalized dataset with trimmed time dimension.
    """
    if "frames" not in ds or ds["frames"].dims != ("y", "x", "time"):
        raise ValueError("ds must contain 'frames' with dims ('y', 'x', 'time')")
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")

    normalization = xr.apply_ufunc(
        fppa.run_norm,
        ds["frames"],
        radius,
        ds["time"],
        input_core_dims=[["time"], [], ["time"]],
        output_core_dims=[["time"], ["time"]],
        exclude_dims={"time"},
        vectorize=True,
    )

    frames = np.where(np.isfinite(normalization[0].data), normalization[0].data, 0)
    new_frames = xr.DataArray(
        frames,
        dims=ds["frames"].dims,
        coords={
            "R": (["y", "x"], ds.R.values),
            "Z": (["y", "x"], ds.Z.values),
            "time": (["time"], normalization[1].data[0, 0, :]),
        },
        attrs=ds["frames"].attrs,
    )
    ds = ds.drop_vars("frames").drop_dims("time")
    ds["frames"] = new_frames

    return ds
