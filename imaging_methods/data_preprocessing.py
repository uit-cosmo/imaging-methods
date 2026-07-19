import fppanalysis as fppa
import xarray as xr
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree

__all__ = ["interpolate_nans_3d", "run_norm_ds"]


def interpolate_nans_3d(ds, mask=None):
    """
    Replace NaN values in a 3D xarray dataset with linear interpolation along spatial dimensions.
    The NaN mask is taken from the first time step and assumed to hold for all times
    (dead pixels are dead for the whole record). The input dataset is not modified.
    The spatial Delaunay triangulation is built once and reused for all time steps.

    Parameters:
        ds (xr.Dataset): Input dataset with 'frames' variable (dims: y, x, time).
        mask (np.ndarray): Boolean array (dims: y, x) specifying if a given pixel should be interpolated, if not none
        all nan pixels OR all pixels in the mask are interpolated.

    Returns:
        xr.Dataset: New dataset with NaNs interpolated.
    """
    # Input validation
    if "frames" not in ds or ds["frames"].dims != ("y", "x", "time"):
        raise ValueError(f"ds must contain 'frames' with dims ('y', 'x', 'time')")
    if ds["frames"].isnull().all():
        raise ValueError("All frame values are NaN")

    ds = ds.copy()
    frames = ds["frames"].values.copy()  # Shape: (ny, nx, nt)
    ny, nx, nt = frames.shape

    # Get the NaN mask (assumed constant across time, taken at t=0)
    nan_mask = np.isnan(frames[:, :, 0])
    if mask is not None:
        nan_mask = nan_mask | mask
    if nan_mask.all():
        raise ValueError("All pixels are NaN in the first time step")
    if not nan_mask.any():
        return ds

    # Coordinate setup for all points
    y_indices, x_indices = np.indices((ny, nx))
    points = np.vstack((y_indices.ravel(), x_indices.ravel())).T  # Shape: (ny*nx, 2)
    valid_mask = ~nan_mask
    valid_points = points[valid_mask.ravel()]  # Coordinates of non-NaN points
    nan_points = points[nan_mask.ravel()]  # Coordinates of NaN points

    # Build the triangulation once; the interpolators only swap values per frame
    triangulation = Delaunay(valid_points)
    # Points outside the convex hull fall back to their nearest valid neighbor
    outside = triangulation.find_simplex(nan_points) < 0
    if outside.any():
        _, nearest_idx = cKDTree(valid_points).query(nan_points[outside])

    # Process each time step
    for t in range(nt):
        valid_values = frames[:, :, t][valid_mask]  # Values at non-NaN points
        interpolator = LinearNDInterpolator(
            triangulation, valid_values, fill_value=np.nan
        )
        interpolated_values = interpolator(nan_points)
        if outside.any():
            interpolated_values[outside] = valid_values[nearest_idx]
        frames[:, :, t][nan_mask] = interpolated_values

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

    # apply_ufunc(vectorize=True) loops over pixels in Python; acceptable for
    # current grid sizes, revisit if it becomes a bottleneck.
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
