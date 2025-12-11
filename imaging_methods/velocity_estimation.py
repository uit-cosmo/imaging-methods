import xarray as xr
import numpy as np
from scipy.signal import windows
from .utils import restrict_to_largest_true_subarray


def get_velocity_from_position(position_da, window_size=3, window_type="boxcar"):
    """
    Compute the velocity signal from a position signal. First the position signal is filtered with a specified
    window_type, then the velocity is computed with a centered difference method.

    Parameters:
    - com_da (xr.DataArray): Position DataArray with dims ('time', 'coord') and coords 'R', 'Z'.
    - window_size (int): Size of the smoothing window (default: 3).
    - window_type (str): Type of smoothing window ('boxcar', 'gaussian', 'hamming', 'blackman', 'triang').

    Returns:
    - velocity_da (xr.DataArray): Velocity with dims ('time', 'coord'), cropped to valid time points.
    """
    if window_size < 1 or not isinstance(window_size, int):
        raise ValueError("window_size must be a positive integer")
    if window_type not in ["boxcar", "gaussian", "hamming", "blackman", "triang"]:
        raise ValueError(
            "window_type must be 'boxcar', 'gaussian', 'hamming', 'blackman', or 'triang'"
        )
    if len(position_da.time) < 2:
        raise ValueError("At least two time points are required")

    # Interpolate NaNs
    com_interp = position_da.interpolate_na(
        dim="time", method="nearest", fill_value="extrapolate"
    )
    com_np = com_interp.values  # Shape: (n_times, 2)

    # Define window parameters
    half_window = window_size // 2
    n_times = len(position_da.time)
    start = half_window + 1
    end = n_times - half_window - 1

    # Check if there are enough points
    if start >= end:
        raise ValueError(
            f"Not enough time points to compute velocity with window_size={window_size}"
        )

    # Generate window using scipy.signal.windows
    if window_type == "gaussian":
        window = windows.gaussian(window_size, std=window_size / 4, sym=True)
    else:
        window = getattr(windows, window_type)(window_size, sym=True)
    window /= window.sum()  # Normalize

    # Compute moving average
    com_smooth = np.array(
        [np.convolve(com_np[:, i], window, mode="same") for i in range(2)]
    ).T

    # Compute time step
    dt = float(position_da.time[1] - position_da.time[0])

    # Compute velocity using central differences
    valid_times = position_da.time[start:end]
    velocity_values = (
        com_smooth[start + 1 : end + 1, :] - com_smooth[start - 1 : end - 1, :]
    ) / (2 * dt)

    # Create output DataArray
    velocity_da = xr.DataArray(
        velocity_values,
        dims=("time", "coord"),
        coords={"time": valid_times, "coord": position_da.coord},
        attrs={
            "description": "Velocity of the contour center of mass in (r, z) coordinates",
            "method": "Finite difference (central) on smoothed COM data",
            "smoothing": f"{window_type.capitalize()} window with window_size={window_size} time steps",
            "units": "Same as com_da units per time unit",
        },
    )

    return velocity_da


def get_averaged_velocity_from_position(
    position_da: xr.DataArray, mask, window_size=3, window_type="boxcar"
):
    """
    Estimates an averaged velocity from a position signal. The position signal is first filtered with the provided
    window and a velocity signal is computed with a centered difference method. The result is the velocity averaged on
    the provided mask.

    Note: The mask should be computed on the condition that the position is close enough to the reference pixel and/or
    that the underlying signal from which the velocity is computed is coherent enough.
    :param position_da:
    :param mask:
    :param window_size:
    :param window_type:
    :return: v, w: Velocity components
    """
    velocity_ds = get_velocity_from_position(position_da, window_size, window_type)
    valid_times = position_da.time[mask]  # DataArray with wanted times
    common_times = valid_times[valid_times.isin(velocity_ds.time)]
    v, w = velocity_ds.sel(time=common_times).mean(dim="time", skipna=True).values
    return v, w


def is_position_near_reference(average_ds, position_da, distance):
    refx, refy = average_ds["refx"].item(), average_ds["refy"].item()

    distances_vector = position_da - [
        average_ds.R.isel(x=refx, y=refy).item(),
        average_ds.Z.isel(x=refx, y=refy).item(),
    ]
    distances = np.sqrt((distances_vector**2).sum(dim="coord"))
    return distances < distance


def get_combined_mask(
    average_ds: xr.Dataset,
    position_da: xr.DataArray,
    extra_condition: xr.DataArray,
    distance: float,
):
    """
    Returns a boolean DataArray indicating time points where:
      - the tracked position is close to the reference position (within `distance`)
      - AND the extra_condition is True

    Then restricts to the longest contiguous block of True values.

    Parameters
    ----------
    average_ds : xr.Dataset:
        Underlying data from which the position is computed
    position_da : xr.DataArray
        DataArray with dims (time, xy), values = [R, Z] displacements or coordinates
    extra_condition : xr.DataArray
        Boolean DataArray with same time dim as position_da
    distance : float
        Maximum allowed distance from reference position (same units as position_da)

    Returns
    -------
    mask : xr.DataArray (bool, along time)
        1D boolean mask with only the longest contiguous valid segment set to True
    """
    near_positions = is_position_near_reference(average_ds, position_da, distance)
    mask = near_positions & extra_condition

    # Restrict to the single longest contiguous block of True values
    return restrict_to_largest_true_subarray(mask)


def get_average_velocity_for_near_com(
    average_ds, contour_ds, velocity_ds, distance, extra_mask=None
):
    refx, refy = average_ds["refx"].item(), average_ds["refy"].item()

    distances_vector = contour_ds.center_of_mass.values - [
        average_ds.R.isel(x=refx, y=refy).item(),
        average_ds.Z.isel(x=refx, y=refy).item(),
    ]
    distances = np.sqrt((distances_vector**2).sum(axis=1))
    mask = distances < distance
    if extra_mask is not None:
        mask = np.logical_and(mask, extra_mask)
    valid_times = contour_ds.time[mask]  # DataArray with wanted times
    common_times = valid_times[valid_times.isin(velocity_ds.time)]

    v_c, w_c = velocity_ds.sel(time=common_times).mean(dim="time", skipna=True).values

    return v_c, w_c
