import xarray as xr
import numpy as np
from scipy.signal import windows
from .utils import restrict_to_largest_true_subarray


def get_velocity_from_position(position_da):
    """
    Compute the velocity signal from a position signal.

    Parameters:
    - position_da (xr.DataArray): Position DataArray with dims ('time', 'coord') and coords 'R', 'Z'.

    Returns:
    - velocity_da (xr.DataArray): Velocity with dims ('time', 'coord'), cropped to valid time points.
    """

    # Compute time step
    dt = float(position_da.time[1] - position_da.time[0])

    # Compute velocity using central differences
    velocity_values = (position_da[2:, :].values - position_da[:-2, :].values) / (
        2 * dt
    )

    # Create output DataArray
    velocity_da = xr.DataArray(
        velocity_values,
        dims=("time", "coord"),
        coords={"time": position_da.time[1:-1], "coord": position_da.coord},
    )

    return velocity_da


def get_averaged_velocity_from_position(position_da: xr.DataArray, mask):
    """
    Estimates an averaged velocity from a position signal. The position signal is first filtered with the provided
    window and a velocity signal is computed with a centered difference method. The result is the velocity averaged on
    the provided mask.

    Note: The mask should be computed on the condition that the position is close enough to the reference pixel and/or
    that the underlying signal from which the velocity is computed is coherent enough.
    :param position_da: Data array with the position components
    :param mask:
    :return: v, w: Velocity components
    """
    velocity_ds = get_velocity_from_position(position_da)
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
