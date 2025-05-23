import fppanalysis as fppa
import xarray as xr
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from phantom import get_t_start_end


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

    ds_normalized = xr.Dataset(
        data_vars={"frames": (["y", "x", "time"], normalization[0].data)},
        coords={"time": normalization[1].data[0, 0, :]},
        attrs=ds.attrs.copy(),
    )
    ds_normalized["frames"].attrs = ds["frames"].attrs.copy()

    return ds_normalized


def load_data_and_preprocess(
    shot, window=None, data_folder="data", preprocessed=True, radius=1000
):
    """
    Load and preprocess APD data for a given shot.
    Parameters:
        shot (int): Shot number.
        window (float, optional): Duration (in seconds) to trim data, centered on mean time.
        data_folder (str): Directory containing data files (default: 'data').
        preprocessed (bool): If True, load preprocessed data; else, apply preprocessing.
        radius (int): Radius for running normalization (window size = 2*radius+1).

    Returns:
        xr.Dataset: Preprocessed APD dataset with normalized frames and interpolated NaNs.
    """
    import os

    file_name = os.path.join(
        data_folder, f"apd_{shot}_preprocessed.nc" if preprocessed else f"apd_{shot}.nc"
    )
    try:
        ds = xr.open_dataset(file_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {file_name} not found")
    except Exception as e:
        raise ValueError(f"Failed to load {file_name}: {str(e)}")

    if window is not None:
        t_start, t_end = get_t_start_end(ds)
        print(f"Data with times from {t_start} to {t_end}")
        t_start = (t_start + t_end) / 2 - window / 2
        t_end = t_start + window
        ds = ds.sel(time=slice(t_start, t_end))

    if preprocessed:
        return ds

    ds = run_norm_ds(ds, radius)
    ds = interpolate_nans_3d(ds)

    # Optionally save preprocessed data
    # ds.to_netcdf(os.path.join(data_folder, f"apd_{shot}_preprocessed.nc"))

    return ds
