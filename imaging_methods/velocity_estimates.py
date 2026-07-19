import numpy as np
import xarray as xr
import velocity_estimation as ve

from .cond_av import find_events_and_2dca
from .contours import get_contour_evolution
from .maximum_trajectory import compute_maximum_trajectory_da
from .method_parameters import MethodParameters, PositionFilterParams
from .utils import get_dr, get_dt, restrict_to_largest_true_subarray, smooth_da

__all__ = [
    "get_velocity_from_position",
    "get_averaged_velocity_from_position",
    "is_position_near_reference",
    "get_combined_mask",
    "get_averaged_velocity",
    "estimate_velocities_synthetic_ds",
]


def get_velocity_from_position(position_da):
    """
    Compute the velocity signal from a position signal.

    Parameters:
    - position_da (xr.DataArray): Position DataArray with dims ('time', 'coord') and coords 'R', 'Z'.

    Returns:
    - velocity_da (xr.DataArray): Velocity with dims ('time', 'coord'), cropped to valid time points.
    """

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
    :return: v, w: Velocity components. (nan, nan) if the mask selects no valid times.
    """
    velocity_ds = get_velocity_from_position(position_da)
    valid_times = position_da.time[mask]  # DataArray with wanted times
    common_times = valid_times[valid_times.isin(velocity_ds.time)]
    if common_times.size == 0:
        return np.nan, np.nan
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
    variable: str,
    position_da: xr.DataArray,
    position_filter: PositionFilterParams,
):
    """
    Returns a boolean DataArray indicating time points where:
      - the tracked position is within a distance, mask_distance from the reference position
      - AND the spatial maximum of average_ds_field is larger than mask_signal_factor times the overrall maximum of
      average_ds_field

    Then restricts to the longest contiguous block of True values.

    Parameters
    ----------
    average_ds : xr.Dataset:
        Underlying data from which the position is computed
    variable: str
        Field name of average_ds to be used ("cross_corr" or "cond_av")
    position_da : xr.DataArray
        DataArray with dims (time, xy), values = [R, Z] displacements or coordinates
    position_filter: PositionFilterParams
        Parameters to compute the mask, including mask_distance and mask_signal_factor.

    Returns
    -------
    mask : xr.DataArray (bool, along time)
        1D boolean mask with only the longest contiguous valid segment set to True
    """
    # mask_distance (in pixels) is converted to a physical distance with the
    # mean radial spacing, assuming approximately square pixels (dz ~ dr).
    distance = position_filter.mask_distance * get_dr(average_ds)
    near_positions = is_position_near_reference(average_ds, position_da, distance)

    signal_high = (
        average_ds[variable].max(dim=["x", "y"]).values
        > position_filter.mask_signal_factor * average_ds[variable].max().item()
    )

    mask = near_positions & signal_high

    # Restrict to the single longest contiguous block of True values
    return restrict_to_largest_true_subarray(mask)


def get_averaged_velocity(
    average_ds, variable, method_parameters, position_method="contouring"
):
    if position_method == "contouring":
        position_da = get_contour_evolution(
            average_ds[variable],
            method_parameters.contouring.threshold_factor,
            max_displacement_threshold=None,
        ).centroid
    elif position_method == "max":
        position_da = compute_maximum_trajectory_da(
            average_ds, variable, method="parabolic"
        )
    else:
        raise NotImplementedError

    position_da, start, end = smooth_da(
        position_da, method_parameters.position_filter, return_start_end=True
    )

    mask = get_combined_mask(
        average_ds.isel(time=slice(start, end)),
        variable,
        position_da,
        method_parameters.position_filter,
    )

    v, w = get_averaged_velocity_from_position(
        position_da=position_da,
        mask=mask,
    )
    return v, w


def estimate_velocities_synthetic_ds(ds, method_parameters: MethodParameters):
    """
    Estimates blob velocities from a given synthetic data dataset and parameters given in method_parameters.
    """
    dt = get_dt(ds)

    events, average_ds = find_events_and_2dca(ds, method_parameters.two_dca)

    v_2dca, w_2dca = get_averaged_velocity(
        average_ds, "cond_av", method_parameters, position_method="contouring"
    )
    v_2dcc, w_2dcc = get_averaged_velocity(
        average_ds, "cross_corr", method_parameters, position_method="contouring"
    )

    v_2dca_max, w_2dca_max = get_averaged_velocity(
        average_ds, "cond_av", method_parameters, position_method="max"
    )
    v_2dcc_max, w_2dcc_max = get_averaged_velocity(
        average_ds, "cross_corr", method_parameters, position_method="max"
    )

    eo = ve.EstimationOptions()
    eo.cc_options.cc_window = method_parameters.two_dca.window * dt
    eo.cc_options.minimum_cc_value = 0
    eo.cc_options.running_mean = False
    try:
        pd = ve.estimate_velocities_for_pixel(
            method_parameters.two_dca.refx,
            method_parameters.two_dca.refy,
            ve.CModImagingDataInterface(ds),
        )
        vx, vy = pd.vx, pd.vy
    except ValueError:
        vx, vy = np.nan, np.nan

    return (
        v_2dca,
        w_2dca,
        v_2dcc,
        w_2dcc,
        v_2dca_max,
        w_2dca_max,
        v_2dcc_max,
        w_2dcc_max,
        vx,
        vy,
    )
